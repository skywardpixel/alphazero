from typing import List, Dict, Any

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.nn_modules.loss_function import AlphaZeroLoss
from alphazero.alphazero.types import TrainExample


class AlphaZeroDataset(Dataset):
    def __init__(self, data: List[TrainExample]) -> None:
        self.data = data

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class NeuralNetTrainer:
    def __init__(self,
                 model: AlphaZeroNeuralNet,
                 config: Dict[str, Any]) -> None:
        self.model = model
        self.loss = AlphaZeroLoss()
        self.batch_size = config['batch_size']
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.optimizer = SGD(self.model.parameters(),
                             lr=config['lr'], momentum=config['momentum'])

    def train(self, data: List[TrainExample]):
        dataset = AlphaZeroDataset(data)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        trainer = create_supervised_trainer(self.model, self.optimizer, self.loss)
        evaluator = create_supervised_evaluator(self.model, metrics={'loss': Loss(self.loss)})

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(data_loader)
            metrics = evaluator.state.metrics
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

        trainer.run(data_loader)
