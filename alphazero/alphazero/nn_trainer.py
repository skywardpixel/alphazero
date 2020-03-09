import logging
from typing import List, Dict, Any

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.nn_modules.loss_function import AlphaZeroLoss
from alphazero.alphazero.types import TrainExample

logger = logging.getLogger(__name__)


class AlphaZeroDataset(Dataset):
    def __init__(self, data: List[TrainExample]) -> None:
        self.data = data

    def __getitem__(self, index: int):
        s, pi, z = self.data[index]
        return s, (pi, z)

    def __len__(self):
        return len(self.data)


class NeuralNetTrainer:
    def __init__(self,
                 model: AlphaZeroNeuralNet,
                 config: Dict[str, Any]) -> None:
        self.model = model
        self.loss = AlphaZeroLoss()
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.max_epochs = config['max_epochs']
        self.optimizer = SGD(self.model.parameters(),
                             lr=config['lr'],
                             momentum=config['momentum'],
                             weight_decay=config['weight_decay'])

    def train(self, data: List[TrainExample], iteration: int):
        dataset = AlphaZeroDataset(data)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        trainer = create_supervised_trainer(self.model,
                                            self.optimizer,
                                            self.loss,
                                            device=self.device)
        evaluator = create_supervised_evaluator(self.model,
                                                metrics={'loss': Loss(self.loss)},
                                                device=self.device)

        # pylint: disable=unused-variable
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch_training_results(engine):  # pylint: disable=unused-argument
            evaluator.run(data_loader)
            metrics = evaluator.state.metrics
            logger.info("Training Results - Epoch: {}  Avg loss: {:.2f}"
                        .format(trainer.state.epoch, metrics['loss']))

        @trainer.on(Events.COMPLETED)
        def log_iteration_training_results(engine):  # pylint: disable=unused-argument
            evaluator.run(data_loader)
            metrics = evaluator.state.metrics
            logger.info("Iteration Training Results - Avg loss: {:.2f}"
                        .format(metrics['loss']))
            self.writer.add_scalar("train/loss", metrics['loss'], iteration)

        trainer.run(data_loader, max_epochs=self.max_epochs)
