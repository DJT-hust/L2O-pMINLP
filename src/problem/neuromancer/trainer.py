"""
Training pipeline
"""

import time
from pathlib import Path

import copy
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

class trainer:
    def __init__(self, components, loss_fn, optimizer, epochs=100, growth_rate=1,
                 patience=5, warmup=0, validate_every=125,
                 clip=100, loss_key="loss", scheduler=None, device="cpu",
                 train_eval_batches=8,
                 loss_report_offset=0.0,
                 tensorboard=False, tb_logdir=None, tb_run_name="run"):
        """
        Initialize the Trainer class.
        """
        self.components = components
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.growth_rate = growth_rate
        self.patience = patience
        self.warmup = warmup
        self.validate_every = max(1, int(validate_every))
        self.clip = clip
        self.loss_key = loss_key
        self.device = device
        self.train_eval_batches = max(0, int(train_eval_batches))
        self.loss_report_offset = float(loss_report_offset)
        self.early_stop_counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.tensorboard = bool(tensorboard)
        self.tb_logdir = tb_logdir
        self.tb_run_name = tb_run_name
        self.writer = None
        self.loss_history = {
            "iters": [],
            "train_eval": [],
            "val": [],
            "test": [],
        }

        if self.tensorboard:
            if SummaryWriter is None:
                print("[TensorBoard] Disabled: torch.utils.tensorboard is unavailable.")
                self.tensorboard = False
            else:
                base_dir = Path(tb_logdir) if tb_logdir else Path("runs")
                run_dir = base_dir / tb_run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(run_dir))
                print(f"[TensorBoard] Logging to {run_dir}")

    def _apply_loss_offset(self, loss_val):
        if loss_val != loss_val:
            return float("nan")
        return float(loss_val) + self.loss_report_offset

    def train(self, loader_train, loader_dev, loader_test=None):
        """
        Perform training with early stopping.
        """
        # init iter
        iters = 0
        stop_training = False
        # initial validation loss calculation
        self.components.eval()
        with torch.no_grad():
            val_loss = self.best_loss = self.calculate_loss(loader_dev)
            test_loss = self.calculate_loss(loader_test) if loader_test is not None else float("nan")
            train_eval_loss = self.calculate_loss(
                loader_train,
                max_batches=self.train_eval_batches if self.train_eval_batches > 0 else None,
            )
        print(
            f"Epoch 0, Iters {iters}, Train Eval Loss: {train_eval_loss:.2f}, "
            f"Validation Loss: {val_loss:.2f}"
        )
        if self.writer is not None:
            self.writer.add_scalar("loss/train_eval", train_eval_loss, iters)
            self.writer.add_scalar("loss/val", val_loss, iters)
            if loader_test is not None:
                self.writer.add_scalar("loss/test", test_loss, iters)
            self.writer.add_scalar("penalty/weight", float(self.loss_fn.penalty_weight), iters)
            self.writer.add_scalar("lr", float(self.optimizer.param_groups[0]["lr"]), iters)
        # training loop
        tick = time.time()
        for epoch in range(self.epochs):
            # early stop
            if stop_training:
                break
            # go through data
            for data_dict in loader_train:
                # training phase
                self.components.train()
                # move to device
                for key in data_dict:
                    if torch.is_tensor(data_dict[key]):
                        data_dict[key] = data_dict[key].to(self.device)
                # forwad pass
                for comp in self.components:
                    data_dict.update(comp(data_dict))
                data_dict = self.loss_fn(data_dict)
                # backward pass
                train_loss = data_dict[self.loss_key] + self.loss_report_offset
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.components.parameters(), self.clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                iters += 1
                if iters % self.validate_every == 0:
                    stop_training = self.validate(
                        epoch,
                        iters,
                        loader_train,
                        loader_dev,
                        loader_test,
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("penalty/weight", float(self.loss_fn.penalty_weight), iters)
                        self.writer.add_scalar("lr", float(self.optimizer.param_groups[0]["lr"]), iters)
                    # update penalty weight
                    self.loss_fn.penalty_weight *= self.growth_rate
                    # early stop
                    if stop_training:
                        break
        tock = time.time()
        elapsed = tock - tick
        print("Training complete.")
        print(f"The training time is {elapsed:.2f} sec.")
        self._save_loss_curve()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def validate(self, epoch, iters, loader_train, loader_dev, loader_test):
        """
        validation
        """
        # validation phase
        self.components.eval()
        with torch.no_grad():
            # use orignal penalty weight for validation
            #self.loss_fn.penalty_weight, temp_weight = self.orig_weight, self.loss_fn.penalty_weight
            # get loss and components
            train_eval_loss = self.calculate_loss(
                loader_train,
                max_batches=self.train_eval_batches if self.train_eval_batches > 0 else None,
            )
            val_loss = self.calculate_loss(loader_dev)
            test_loss = self.calculate_loss(loader_test) if loader_test is not None else float("nan")
            
            # also compute objective and violation separately for analysis
            val_obj, val_viol = self._get_obj_and_viol(loader_dev) if loader_dev is not None else (float("nan"), float("nan"))
            test_obj, test_viol = self._get_obj_and_viol(loader_test) if loader_test is not None else (float("nan"), float("nan"))
            
            if loader_test is not None:
                print(
                    f"Epoch {epoch}, Iters {iters}, Train Eval Loss: {train_eval_loss:.2f}, "
                    f"Validation Loss: {val_loss:.2f} (obj={val_obj:.2f}, viol={val_viol:.2f}), "
                    f"Test Loss: {test_loss:.2f} (obj={test_obj:.2f}, viol={test_viol:.2f})"
                )
            else:
                print(
                    f"Epoch {epoch}, Iters {iters}, Train Eval Loss: {train_eval_loss:.2f}, "
                    f"Validation Loss: {val_loss:.2f}"
                )

            self.loss_history["iters"].append(iters)
            self.loss_history["train_eval"].append(float(train_eval_loss))
            self.loss_history["val"].append(float(val_loss))
            self.loss_history["test"].append(float(test_loss))

            if self.writer is not None:
                self.writer.add_scalar("loss/train_eval", train_eval_loss, iters)
                self.writer.add_scalar("loss/val", val_loss, iters)
                if loader_test is not None:
                    self.writer.add_scalar("loss/test", test_loss, iters)
                self.writer.add_scalar("lr", float(self.optimizer.param_groups[0]["lr"]), iters)
            # restore weight
            #self.loss_fn.penalty_weight = temp_weight
        # turn into training phase
        self.components.train()
        # start early stop after warmup
        if iters // self.validate_every >= self.warmup:
            # early stopping update
            self.update_early_stopping(val_loss)
            # check patience condition
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at iters {iters}")
                # load best model
                if self.best_model_state is not None:
                    self.components.load_state_dict(self.best_model_state)
                    print("Best model loaded.")
                return True
            else:
                return False

    def calculate_loss(self, loader, max_batches=None):
        """
        Calculate loss for a given dataset loader.
        """
        if loader is None:
            return float("nan")
        total_loss = 0.0
        num_batches = 0
        for batch_idx, data_dict in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            # move to device
            for key in data_dict:
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key].to(self.device)
            # forward pass
            for comp in self.components:
                data_dict.update(comp(data_dict))
            # get loss components
            result_dict = self.loss_fn(data_dict)
            total_loss += (result_dict[self.loss_key] + self.loss_report_offset).item()
            num_batches += 1
        if num_batches == 0:
            return float("nan")
        return total_loss / num_batches

    def _get_obj_and_viol(self, loader):
        """
        Compute average objective and violation separately for diagnostics.
        """
        if loader is None:
            return float("nan"), float("nan")
        total_obj = 0.0
        total_viol = 0.0
        for data_dict in loader:
            # move to device
            for key in data_dict:
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key].to(self.device)
            # forward pass
            for comp in self.components:
                data_dict.update(comp(data_dict))
            # get obj and viol
            if hasattr(self.loss_fn, 'cal_obj'):
                total_obj += self.loss_fn.cal_obj(data_dict).mean().item()
            if hasattr(self.loss_fn, 'cal_constr_viol'):
                total_viol += self.loss_fn.cal_constr_viol(data_dict).mean().item()
        n = len(loader)
        return total_obj / n, total_viol / n

    def update_early_stopping(self, val_loss):
        """
        Update the early stopping counter and model state.
        """
        # update with better loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(self.components.state_dict())
            self.early_stop_counter = 0  # reset early stopping counter
        else:
            self.early_stop_counter += 1

    def _save_loss_curve(self):
        """
        Save train/val/test loss curves as a PNG file.
        """
        if len(self.loss_history["iters"]) == 0:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("[LossCurve] Skip plotting: matplotlib is not available.")
            return

        base_dir = Path(self.tb_logdir) if self.tb_logdir else Path("runs")
        curve_dir = base_dir / "loss_curves"
        curve_dir.mkdir(parents=True, exist_ok=True)
        curve_path = curve_dir / f"{self.tb_run_name}_loss_curve.png"

        eps = 1e-12
        iters = self.loss_history["iters"]
        train_eval_loss = [max(float(x), eps) for x in self.loss_history["train_eval"]]
        val_loss = [max(float(x), eps) for x in self.loss_history["val"]]
        plt.figure(figsize=(9, 5))
        plt.plot(iters, train_eval_loss, label="train_eval_loss", linewidth=2)
        plt.plot(iters, val_loss, label="val_loss", linewidth=2)

        has_test = any(x == x for x in self.loss_history["test"])
        if has_test:
            test_loss = [max(float(x), eps) if x == x else float("nan") for x in self.loss_history["test"]]
            plt.plot(iters, test_loss, label="test_loss", linewidth=2)

        plt.xlabel("Iters")
        plt.ylabel("Loss (log scale)")
        plt.yscale("log")
        plt.title("Training / Validation / Test Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(curve_path, dpi=150)
        plt.close()

        print(f"[LossCurve] Saved: {curve_path}")
