import copy
import torch
import torch.nn.functional as F
import tqdm

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils


def kl_loss(student_logits, teacher_logits, temperature):
    teacher_out = F.softmax(teacher_logits / temperature, dim=1)
    student_out = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_out, teacher_out, reduction='batchmean') * (temperature ** 2)


@timer
def scrub(
    ori_model,
    train_remain_loader,
    train_forget_loader,
    unlearn_epoch,
    unlearn_rate,
    logger,
    console_handler,
    loader_dict,
    experiment_path,
    KL_temperature=4,
    eval_opt=eval_opt,
    disable_bn=False
):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }

    log_utils.enable_console_logging(logger, console_handler, False)

    for epoch in tqdm.trange(unlearn_epoch):
        unlearn_model.train()

        # 1) forget phase: teacher와 멀어지기
        for x, y in train_forget_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(x)

            student_logits = unlearn_model(x)
            loss = -kl_loss(student_logits, teacher_logits, KL_temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2) remain phase: teacher를 따라가기
        for x, y in train_remain_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(x)

            student_logits = unlearn_model(x)
            loss = kl_loss(student_logits, teacher_logits, KL_temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            if key in cur_accs_dict:
                accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch + 1, accs_dict, experiment_path)

    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model