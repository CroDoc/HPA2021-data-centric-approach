import torch.nn as nn
import torch
import torch.nn.functional as F
import os

LABEL_COUNT = 19

class MagicValidator():
    def __init__(self, dataset, validation_loader, criterion, device, calculate_device = 'cpu'):
        self.dataset = dataset
        self.validation_loader = validation_loader
        self.label_map = self.dataset.label_map

        self.criterion = criterion
        self.best_score = None
        self.device = device
        self.calculate_device = calculate_device

        self.label_score = {}
        self.criterion_score = {}
    
    def validate(self, net):
        net.eval()

        loss_per_label = [[] for x in range(LABEL_COUNT)]
        acc_loss_per_label = [[] for x in range(LABEL_COUNT)]
        loss_per_output_label = [[] for x in range(len(self.label_map) + 1)]

        confidence_and_gt = {x:[] for x in self.label_map}
        AP = {x:0 for x in self.label_map}

        total_loss, epoch_size = 0, 0

        with torch.no_grad():
            for data in self.validation_loader:

                inputs, labels, indices = data
                inputs = inputs.to(self.device)
                
                labels = labels.to(self.calculate_device)
                outputs = net(inputs).to(self.calculate_device)
                
                loss = self.criterion(outputs, labels)

                total_loss += loss.mean(dim=1).sum().item()
                epoch_size += outputs.size(0)

                accuracy_losses = torch.zeros_like(outputs)
                
                accuracy_losses[outputs >= 0.5] = 1
                accuracy_losses[outputs < 0.5] = 0

                accuracy_losses = torch.abs(outputs - labels)

                for batch_index in range(len(indices)):

                    dataset_index = indices[batch_index].item()
                    metadata = self.dataset.image_metadata[dataset_index]

                    # mAP loss
                    for output_label in self.label_map:
                        if output_label in metadata.metric_labels:
                            output_index = self.label_map[output_label]
                            confidence_and_gt[output_label].append((outputs[batch_index][output_index].item(), labels[batch_index][output_index].item()))

                            label_loss = loss[batch_index][output_index]
                            loss_per_output_label[output_index].append(label_loss)

                    image_loss = loss[batch_index]
                    image_acc_loss = accuracy_losses[batch_index]

                    for image_label in metadata.metric_labels:

                        if image_label in self.label_map:
                            acc_loss = image_acc_loss[self.label_map[image_label]]
                            label_loss = image_loss[self.label_map[image_label]]
                        else:
                            continue
                        
                        output_label = len(self.label_map)
                        if image_label in self.label_map:
                            output_label = self.label_map[image_label]

                        acc_loss_per_label[image_label].append(acc_loss)
                        loss_per_label[image_label].append(label_loss)

        for x in acc_loss_per_label:
            if not x:
                x.append(torch.tensor([0.0]).to(self.calculate_device))
        for x in loss_per_label:
            if not x:
                x.append(torch.tensor([0.0]).to(self.calculate_device))
        for x in loss_per_output_label:
            if not x:
                x.append(torch.tensor([0.0]).to(self.calculate_device))

        acc_loss_per_label = [1.0 - torch.stack(x).mean().item() for x in acc_loss_per_label]
        loss_per_label = [torch.stack(x).mean().item() for x in loss_per_label]
        loss_per_output_label = [torch.stack(x).mean() for x in loss_per_output_label]

        validation_loss = torch.stack(loss_per_output_label).mean().item()

        loss_per_output_label = [x.item() for x in loss_per_output_label]

        acc_loss_per_label = [int(round(x * 100)) for x in acc_loss_per_label]
        loss_per_label = [round(x, 5) for x in loss_per_label]

        mAP = 0
        zero_positives = 0

        for label in self.label_map:
            label_index = self.label_map[label]
            
            curr_loss = loss_per_output_label[label_index]

            if curr_loss < self.criterion_score.get(label, 100000000000000):
                self.criterion_score[label] = curr_loss

        for label in confidence_and_gt:

            cg = confidence_and_gt[label]
        
            cg.sort(key = lambda x : -x[1])

            MAX_VAL = 0.0
            all_examples = 0
            positives_value = 0
            for i in range(len(cg)):
                all_examples += 1
                positives_value += cg[i][1]
                MAX_VAL += positives_value / all_examples

            cg.sort(key = lambda x : -x[0])

            all_examples = 0
            accuracy_sum = 0
            positives_value = 0
            for i in range(len(cg)):
                all_examples += 1
                positives_value += cg[i][1]

                accuracy_sum += positives_value / all_examples
            
            if positives_value > 0 and MAX_VAL > 0:
                AP[label] = accuracy_sum / MAX_VAL
            else:
                AP[label] = 0
                zero_positives += 1
            mAP += AP[label]

            if AP[label] > self.label_score.get(label, 0):
                self.label_score[label] = AP[label]
        
        if len(self.label_map) - zero_positives == 0:
            mAP = 0
        else:
            mAP /= len(self.label_map) - zero_positives

        classic_loss = total_loss/epoch_size

        loss_per_output_label = [round(x, 2) for x in loss_per_output_label]

        #print(acc_loss_per_label)
        print(loss_per_label)
        print(loss_per_output_label)
        print(sorted(AP.items()))
        print(mAP, classic_loss)

        return mAP, classic_loss

class ModelSaver():
    def __init__(self, target_loss, model_name, save_dir='new-models/', bigger = False):
        self.target_loss = target_loss
        self.save_path = save_dir + model_name
        self.bigger = bigger
    
    def save_model(self, network_wrapper, model_loss, epoch):
        if (not self.bigger and model_loss < self.target_loss) or (self.bigger and model_loss > self.target_loss):
            print('--- BEST ---', model_loss)
            self.target_loss = model_loss
            
        torch.save(network_wrapper.network.state_dict(), self.save_path + '-' + str(epoch) + '.pt')

        with open(self.save_path + '.txt', 'w') as f:
            print(type(network_wrapper).__name__, file=f)
            print(network_wrapper.output_labels, file=f)

    def save_model_if_better(self, network_wrapper, model_loss, epoch):
        if (not self.bigger and model_loss < self.target_loss) or (self.bigger and model_loss > self.target_loss):
            print('--- BEST ---', model_loss)
            self.target_loss = model_loss
            torch.save(network_wrapper.network.state_dict(), self.save_path + '-' + str(epoch) + '.pt')

            with open(self.save_path + '.txt', 'w') as f:
                print(type(network_wrapper).__name__, file=f)
                print(network_wrapper.output_labels, file=f)