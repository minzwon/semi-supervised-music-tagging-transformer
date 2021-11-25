import os


class DirManager:
    """A utility class to manage directories.
    In this way, because IDEs would be aware of attribute names, development can be
    safer (e.g. pre-warning syntax errors) and more convenient (e.g. autocomplete).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        self.tensorboard_dir = os.path.join(output_dir, 'tensorboard')

        self.best_model_onnx_name = 'best_model.onnx'
        self.best_model_statedict = 'best_model_statedict.pt'
        self.best_model_torchscript_name = 'best_model_torchscript.pt'
        if os.path.exists(self.checkpoint_dir) and os.listdir(self.checkpoint_dir) != []:
            raise RuntimeError(
                f'{self.checkpoint_dir} already exists and not blank. '
                f'This is supported in Pytorch-Lightning but it can be very confusing, therefore '
                f'I do not recommended using it in such a way. Use a different path'
                f'to save the experiment results!'
            )
