import torch
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, train_data, criterion, optimizer, epoch_step, epoch, endepoch ):
    train_loss = 0
    model_train = torch.nn.DataParallel(model).cuda()
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{endepoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_data):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
            targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
            optimizer.zero_grad()
            output = model_train(images)
            loss = criterion.forward(targets, output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(**{'train_loss': train_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Finish Train')
    print('Epoch:' + str(epoch + 1) + '/' + str(endepoch))


def validate(model, val_data, criterion, optimizer, epoch_step_val, epoch, endepoch):
    val_loss = 0
    model_train = torch.nn.DataParallel(model).cuda()
    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{endepoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_data):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                out = model_train(images)
                optimizer.zero_grad()
                loss = criterion.forward(targets, out)
                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(epoch))
    torch.save(model.state_dict(), 'weight/trained_weight_2.pth')

