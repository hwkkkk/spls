"""Drop-in replacement for method.ldmu used only by the ablation runner.

The training loop mirrors method/ldmu.py (same deep copies, SGD(momentum=0.9), MSELoss(mean),
per-epoch evaluation and plotting, same call order); only the target construction (and, for the
loss ablation, the loss space) is swapped via spls_ablation.targets.spls_target_and_loss.
Non-finite loss/target aborts the run instead of being patched over.
"""
import copy
import math
import os

import torch
import tqdm
from torch import nn

import log_utils
from method.utils import keys, eval_opt as default_eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders

from spls_ablation.targets import spls_target_and_loss


class NonFiniteError(RuntimeError):
    pass


# filled in by the runner after the call so it can write the run status next to the logs
LAST_EXPERIMENT_PATH = {}


def make_ablation_ldmu(variant):
    def ablation_ldmu(k,
        ori_model, train_forget_loader,
        unlearn_epoch, unlearn_rate,
        logger, console_handler,
        loader_dict, experiment_path,
        soft_label,
        eval_opt=default_eval_opt, disable_bn=False,
    ):
        LAST_EXPERIMENT_PATH["path"] = experiment_path
        logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
        logger.info(f"eval option {eval_opt}")
        logger.info(f"SPLS ablation variant: {variant}")

        unlearn_model = copy.deepcopy(ori_model).to("cuda")
        test_model = copy.deepcopy(ori_model).to("cuda")
        test_model.eval()

        optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

        accs_dict = {key: [] for key in keys}

        # 2026-11-04 coupling diagnostic: per-step grad / update norms (backbone vs head), only if SPLS_GRADLOG is set
        gradlog = os.environ.get("SPLS_GRADLOG")
        if gradlog:
            named = [(n, p) for n, p in unlearn_model.named_parameters() if p.requires_grad]
            is_head = [n.startswith(("head", "fc")) for n, _ in named]
            glog = open(gradlog, "w")
            glog.write("epoch,step,loss,grad_backbone,grad_head,update_backbone,update_head\n")
            gstep = 0

        log_utils.enable_console_logging(logger, console_handler, False)

        for epoch in tqdm.trange(unlearn_epoch):
            epoch_loss = 0.0
            num_batches = 0

            for x, y in train_forget_loader:
                x, y = x.to("cuda"), y.to("cuda")

                unlearn_model.train()

                if disable_bn:
                    for module in unlearn_model.modules():
                        if isinstance(module, nn.BatchNorm2d):
                            module.eval()

                optimizer.zero_grad()

                with torch.no_grad():
                    pred_label = test_model(x).clone()
                expected_classes = len(train_forget_loader.dataset.classes)
                if pred_label.ndim != 2 or pred_label.shape[1] != expected_classes:
                    raise ValueError(
                        f"Expected {expected_classes} class logits, got {tuple(pred_label.shape)}. "
                        "Check the model classification forward before running ablation."
                    )

                ori_logits = unlearn_model(x)
                target_label, loss = spls_target_and_loss(ori_logits, pred_label, y, variant)
                if not torch.isfinite(target_label).all():
                    logger.error(f"NONFINITE target at epoch {epoch+1} batch {num_batches+1}")
                    raise NonFiniteError("non-finite target")
                if not torch.isfinite(loss):
                    logger.error(f"NONFINITE loss at epoch {epoch+1} batch {num_batches+1}: {loss.item()}")
                    raise NonFiniteError("non-finite loss")

                loss.backward()
                if gradlog:
                    before = [p.detach().clone() for _, p in named]
                    gn = [(p.grad.detach().float() ** 2).sum() if p.grad is not None else torch.zeros((), device="cuda")
                          for _, p in named]
                optimizer.step()
                if gradlog:
                    un = [((p.detach() - b).float() ** 2).sum() for (_, p), b in zip(named, before)]
                    part = lambda v, h: torch.stack([x for x, hh in zip(v, is_head) if hh == h]).sum().sqrt().item()
                    glog.write(f"{epoch+1},{gstep},{loss.item():.6g},{part(gn, False):.6g},{part(gn, True):.6g},"
                               f"{part(un, False):.6g},{part(un, True):.6g}\n")
                    gstep += 1
                    del before

                epoch_loss += loss.item()
                num_batches += 1

            logger.info(f"epoch {epoch+1} loss {epoch_loss / num_batches:.4f}")

            cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
            for key in keys:
                accs_dict[key].append(cur_accs_dict[key])
            # 2026-11-03: method.utils skips eval (returns NaN) before the last epoch -> check only the last epoch
            if epoch + 1 == unlearn_epoch and any(math.isnan(cur_accs_dict[key]) for key in keys if eval_opt[key]):
                logger.error(f"NONFINITE accuracy at epoch {epoch+1}")
                raise NonFiniteError("non-finite accuracy")

            plot_unlearn_remain_acc_figure(epoch + 1, accs_dict, experiment_path)

        if gradlog:
            glog.close()
        log_utils.enable_console_logging(logger, console_handler, True)
        return unlearn_model

    return ablation_ldmu
