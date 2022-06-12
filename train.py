import argparse
from time import time
from datetime import timedelta

import torch

from utils import get_train_config, get_logger


def train(logger, epoch, model, device, train_loader, optimizer, lr_scheduler, criterion):
    """Training phase"""
    model.train()
    train_loss = 0 
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            logger.info('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Learning rate: ({})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total, optimizer.param_groups[0]['lr']))
    #lr_scheduler.step()

    return train_loss / (batch_idx + 1), 100. * correct / total


def test(logger, model, device, test_loader, criterion):
    """Training phase"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        outputs = model(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
    logger.info('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    return test_loss / (batch_idx + 1), 100. * correct / total


def main(args):
    config = get_train_config(args) # Get config
    logger = get_logger(config) # Get logger 

    # Get componetns
    model = config['model']
    device, device_ids = config['device'], config['device_ids']
    optimizer = config['optimizer']
    lr_scheduler = config['lr_scheduler']
    train_loader = config['data_loader']
    test_loader = config['valid_loader']
    criterion = config['loss']
    
    # Logging config
    logger.info(f"Experiment (Training Phase): {config['name']}")
    logger.info(f"Model : \n{config['model']}") 
    logger.info(f"Devices : {config['device']}")
    logger.info(f"Devices ID : {config['device_ids']}")
    logger.info(f"See detail of experiment at {config['config_file_path']}")

    # Train with tensorboard
    start_time = time() # Time

    ###################################################################################################################
    ############################################### MODEL LOADING #####################################################
    model.load_state_dict(torch.load("best_model.pth"))
    for param in model.parameters():
        param.requires_grad = True

    #for param in model.efficientnet.classifier.parameters():
    #    param.requires_grad = True
    
    #for param in model.efficientnet.features[4:].parameters():
    #    param.requires_grad = True

    from torch.nn.modules.batchnorm import BatchNorm2d
    def bn_freeze(model):
        if type(model) is BatchNorm2d:
            for param in model.parameters():
                param.requires_grad = False
        for child in model.children():
            bn_freeze(child)
    bn_freeze(model)

    ###################################################################################################################
    ###################################################################################################################


    stop_flag = False
    best_valid_loss = float("inf")
    for epoch in range(config['train']['epochs']):
        epoch_start_time = time()
        try:
            train_loss, train_acc = train(logger, epoch,  model,  device,  train_loader,  optimizer, lr_scheduler, criterion)
        except KeyboardInterrupt:
            print("Menu")
            print("1. Change learning rate")
            print("2. Stop")

            while True:
                ch = int(input())
                if ch == 1:
                    optimizer.param_groups[0]['lr'] = float(input("Learning rate : "))
                    break
                elif ch == 2:
                    stop_flag = True
                    break
        
        if stop_flag:
            valid_loss, valid_acc = test(logger, model, device, test_loader, criterion)
            # Model save
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                torch.save(config['model'].state_dict(), config['model_save_dir'] / f'acc_{valid_acc}_best_model.pth')

            if (epoch + 1) % config['train']['save_period'] == 0:
                torch.save(config['model'].state_dict(), config['model_save_dir'] / f'model_weights_{epoch + 1}epoch.pth')
            break

        valid_loss, valid_acc = test(logger, model, device, test_loader, criterion)
        logger.info("Epoch time : " + str(timedelta(seconds=time() - epoch_start_time)).split(".")[0])

        # Model save
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            torch.save(config['model'].state_dict(), config['model_save_dir'] / f'acc_{int(valid_acc*1000)}_best_model.pth')

        if (epoch + 1) % config['train']['save_period'] == 0:
            torch.save(config['model'].state_dict(), config['model_save_dir'] / f'model_weights_{epoch + 1}epoch.pth')


    # Time
    end_time = time()
    logger.info("Experiment time " + str(timedelta(seconds=end_time - start_time)).split(".")[0])


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="CIFAR10 Classifier")
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')

    # TODO : Unimplemented
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    
    # TODO : TensorboardX Option
    args = parser.parse_args()
    main(args)