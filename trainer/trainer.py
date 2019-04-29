import random

import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from utils.util import denormalize


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, generator, discriminator, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(config, generator, discriminator, loss, metrics, optimizer, lr_scheduler, resume,
                                      train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set models to train mode
        self.generator.train()
        self.discriminator.train()

        total_generator_loss = 0
        total_discriminator_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            blurred = sample['blurred'].to(self.device)
            sharp = sample['sharp'].to(self.device)

            # get G's output
            deblurred = self.generator(blurred)

            # denormalize
            with torch.no_grad():
                denormalized_blurred = denormalize(blurred)
                denormalized_sharp = denormalize(sharp)
                denormalized_deblurred = denormalize(deblurred)

            if batch_idx % 100 == 0:
                # save blurred, sharp and deblurred image
                self.writer.add_image('blurred', make_grid(denormalized_blurred.cpu()))
                self.writer.add_image('sharp', make_grid(denormalized_sharp.cpu()))
                self.writer.add_image('deblurred', make_grid(denormalized_deblurred.cpu()))

            # get D's output
            sharp_discriminator_out = self.discriminator(sharp)
            deblurred_discriminator_out = self.discriminator(deblurred)

            # set critic_updates
            if self.config['loss']['adversarial'] == 'wgan_gp_loss':
                critic_updates = 5
            else:
                critic_updates = 1

            # train discriminator
            discriminator_loss = 0
            for i in range(critic_updates):
                self.discriminator_optimizer.zero_grad()

                # train discriminator on real and fake
                if self.config['loss']['adversarial'] == 'wgan_gp_loss':
                    gp_lambda = self.config['others']['gp_lambda']
                    alpha = random.random()
                    interpolates = alpha * sharp + (1 - alpha) * deblurred
                    interpolates_discriminator_out = self.discriminator(interpolates)
                    kwargs = {
                        'gp_lambda': gp_lambda,
                        'interpolates': interpolates,
                        'interpolates_discriminator_out': interpolates_discriminator_out,
                        'sharp_discriminator_out': sharp_discriminator_out,
                        'deblurred_discriminator_out': deblurred_discriminator_out
                    }
                    wgan_loss_d, gp_d = self.adversarial_loss('D', **kwargs)
                    discriminator_loss_per_update = wgan_loss_d + gp_d

                    self.writer.add_scalar('wgan_loss_d', wgan_loss_d.item())
                    self.writer.add_scalar('gp_d', gp_d.item())
                elif self.config['loss']['adversarial'] == 'gan_loss':
                    kwargs = {
                        'sharp_discriminator_out': sharp_discriminator_out,
                        'deblurred_discriminator_out': deblurred_discriminator_out
                    }
                    gan_loss_d = self.adversarial_loss('D', **kwargs)
                    discriminator_loss_per_update = gan_loss_d

                    self.writer.add_scalar('gan_loss_d', gan_loss_d.item())
                else:
                    # add other loss if you like
                    raise NotImplementedError

                discriminator_loss_per_update.backward(retain_graph=True)
                self.discriminator_optimizer.step()
                discriminator_loss += discriminator_loss_per_update.item()

            discriminator_loss /= critic_updates
            self.writer.add_scalar('discriminator_loss', discriminator_loss)
            total_discriminator_loss += discriminator_loss

            # train generator
            self.generator_optimizer.zero_grad()

            content_loss_lambda = self.config['others']['content_loss_lambda']
            kwargs = {
                'deblurred_discriminator_out': deblurred_discriminator_out
            }
            adversarial_loss_g = self.adversarial_loss('G', **kwargs)
            content_loss_g = self.content_loss(deblurred, sharp) * content_loss_lambda
            generator_loss = adversarial_loss_g + content_loss_g

            self.writer.add_scalar('adversarial_loss_g', adversarial_loss_g.item())
            self.writer.add_scalar('content_loss_g', content_loss_g.item())
            self.writer.add_scalar('generator_loss', generator_loss.item())

            generator_loss.backward()
            self.generator_optimizer.step()
            total_generator_loss += generator_loss.item()

            # calculate the metrics
            total_metrics += self._eval_metrics(denormalized_deblurred, denormalized_sharp)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] generator_loss: {:.6f} discriminator_loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        generator_loss.item(),  # it's a tensor, so we call .item() method
                        discriminator_loss  # just a num
                    )
                )

        log = {
            'generator_loss': total_generator_loss / len(self.data_loader),
            'discriminator_loss': total_discriminator_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        self.generator_lr_scheduler.step()
        self.discriminator_lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.generator.eval()
        self.discriminator.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                blurred = sample['blurred'].to(self.device)
                sharp = sample['sharp'].to(self.device)

                deblurred = self.generator(blurred)
                deblurred_discriminator_out = self.discriminator(deblurred)

                content_loss_lambda = self.config['others']['content_loss_lambda']
                kwargs = {
                    'deblurred_discriminator_out': deblurred_discriminator_out
                }
                adversarial_loss_g = self.adversarial_loss('G', **kwargs)
                content_loss_g = self.content_loss(deblurred, sharp) * content_loss_lambda
                loss_g = adversarial_loss_g + content_loss_g

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('adversarial_loss_g', adversarial_loss_g.item())
                self.writer.add_scalar('content_loss_g', content_loss_g.item())
                self.writer.add_scalar('loss_g', loss_g.item())
                total_val_loss += loss_g.item()

                total_val_metrics += self._eval_metrics(denormalize(deblurred), denormalize(sharp))

        # add histogram of model parameters to the tensorboard
        for name, p in self.generator.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
