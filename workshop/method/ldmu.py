import copy
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils


@timer
def ldmu(k,
    ori_model, train_forget_loader,
    unlearn_epoch, unlearn_rate,
    logger, console_handler,
    loader_dict, experiment_path,
    soft_label,
    eval_opt=eval_opt, disable_bn=False,
):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")

    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    test_model = copy.deepcopy(ori_model).to("cuda")
    test_model.eval()

    print({soft_label})

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }

    log_utils.enable_console_logging(logger, console_handler, False)

    for epoch in tqdm.trange(unlearn_epoch):
        epoch_loss = 0.0
        num_batches = 0

        for x, y in train_forget_loader:
            x, y = x.to("cuda"), y.to("cuda")
            batch_size = x.shape[0]
            idx = torch.arange(batch_size, device=x.device)

            unlearn_model.train()

            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

            optimizer.zero_grad()

            with torch.no_grad():
                pred_label = test_model(x).clone()

            if soft_label == "inf":
                # probability-space MSE
                pred_label[idx, y] = 1e-10
                ori_logits = unlearn_model(x)

                student_prob = F.softmax(ori_logits, dim=1)
                teacher_prob = F.softmax(pred_label, dim=1)

                loss = criterion(student_prob, teacher_prob)

            elif soft_label == "ce_mean":
                ori_logits = unlearn_model(x)

                other_mask = torch.ones_like(ori_logits, dtype=torch.bool)
                other_mask[idx, y] = False

                mse_loss = criterion(ori_logits[other_mask], pred_label[other_mask])
                ce_loss = -F.cross_entropy(ori_logits, y)

                loss = 0.5 * mse_loss + 0.5 * ce_loss

            elif soft_label == "lowest":
                tmp = pred_label.clone()
                pred_label[idx, y] = tmp.min(dim=1).values

                ori_logits = unlearn_model(x)
                loss = criterion(ori_logits, pred_label)
            
            elif soft_label == "lowest-std":
                target_label = pred_label.clone()
                batch_idx = torch.arange(target_label.size(0), device=target_label.device)

                tmp_min = target_label.min(dim=1).values   # [B]
                tmp_std = target_label.std(dim=1)          # [B]

                target_label[batch_idx, y] = tmp_min - tmp_std*k
                

                ori_logits = unlearn_model(x)
                loss = criterion(ori_logits, target_label)

            elif soft_label == "max_remain-std":
                with torch.no_grad():
                    target_label = pred_label.clone()
                    B, C = target_label.shape
                    batch_idx = torch.arange(B, device=target_label.device)

                    remain_mask = torch.ones_like(target_label, dtype=torch.bool)
                    remain_mask[batch_idx, y] = False

                    remain_vals = target_label[remain_mask].view(B, C - 1)
                    tmp_max_remain = remain_vals.max(dim=1).values

                    tmp_std_all = target_label.std(dim=1, unbiased=False)

                    target_label[batch_idx, y] = tmp_max_remain - k * tmp_std_all

                ori_logits = unlearn_model(x)
                loss = criterion(ori_logits, target_label)

            else:
                raise ValueError("Unknown soft label method")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        logger.info(f"epoch {epoch+1} loss {epoch_loss / num_batches:.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch + 1, accs_dict, experiment_path)

    log_utils.enable_console_logging(logger, console_handler, True)
    return unlearn_model