import argparse
from time import time
from datetime import timedelta

import torch

from utils import get_train_config, get_logger

def train(
        logger,
        epoch, 
        model, 
        device, 
        train_loader, 
        optimizer,
        lr_scheduler,
        criterion
    ):
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
            logger.info('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    lr_scheduler.step()

    return train_loss / (batch_idx + 1), 100. * correct / total


def test(
        logger,
        model, 
        device, 
        test_loader, 
        criterion
    ):
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
    model, device, device_ids = config['model'], config['device'], config['device_ids']
    optimizer, lr_scheduler = config['optimizer'], config['lr_scheduler']
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

    ################### MODEL LOADING #########################
    #config['model'].load_state_dict(torch.load("saved/CIFAR10 Classifier/ResNet152Wrapper/train/save_instances/model_weights_best_model_epoch.pth"))
    ###########################################################

    ################### MODEL CUSTOMIZATION ###################
    for param in config['model'].parameters():
        param.requires_grad = False

    for param in config['model'].resnet152.fc.parameters():
        param.requires_grad = True
    
    for param in config['model'].resnet152.layer4.parameters():
        param.requires_grad = True

    for param in config['model'].resnet152.layer3.parameters():
        param.requires_grad = True
    
    for param in config['model'].resnet152.layer2.parameters():
        param.requires_grad = True
    ###########################################################

    for epoch in range(config['train']['epochs']):
        epoch_start_time = time()
        train_loss, train_acc = train(
            logger,
            epoch, 
            model, 
            device, 
            train_loader, 
            optimizer,
            lr_scheduler,
            criterion
        )
        valid_loss, valid_acc = test(
            logger,
            model, 
            device, 
            test_loader, 
            criterion
        )
        logger.info("Epoch time : " + str(timedelta(seconds=time() - epoch_start_time)).split(".")[0])
        # Model save
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