"""Native curriculumagent1.1.2 Torch Senior model; extracted without TF-only imports."""
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class Grid2OpCustomModelTorch(TorchModelV2, nn.Module):
    """
    A custom PyTorch model for policy gradient algorithms in the Grid2Op environment.

    This class defines a custom neural network architecture, and supports loading pretrained models, including
    the Junior and Senior models, for use within the RLlib framework in the Grid2Op environment.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        """
        Constructor for the custom PyTorch model.

        This method initializes the custom model using the observation space, action space, and model configuration 
        from RLlib. It supports layer customization and loading pretrained models, such as Junior models.

        Args:
            obs_space (Box): The observation space provided by RLlib.
            action_space (Discrete): The action space provided by RLlib.
            num_outputs (int): Number of output nodes, corresponding to the action space.
            model_config (dict): Configuration dictionary for the model.
            name (str): Name of the model.
            customized_model_kwargs (dict): Custom model parameters including layer sizes, activation functions, 
                and initializers.

        Returns:
            None.
        """
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name

        if "custom_config" in customized_model_kwargs.keys() and customized_model_kwargs["custom_config"] is not None:
            cconfig = customized_model_kwargs["custom_config"]
        else:
            cconfig = {}
        layer_size, initializer, activation = self._extract_config(cconfig)

        # Define layers
        self.layer1 = nn.Linear(obs_space.shape[0], layer_size[0])
        self.layer2 = nn.Linear(layer_size[0], layer_size[1])
        self.layer3 = nn.Linear(layer_size[1], layer_size[2])
        self.layer4 = nn.Linear(layer_size[2], layer_size[3])
        self.act_layer = nn.Linear(layer_size[3], num_outputs)
        self.val_hidden_layer = nn.Linear(layer_size[3], action_space.n)
        self.val_layer = nn.Linear(action_space.n, 1)

        # Set activation and initializer
        self.activation1 = deepcopy(activation)
        self.activation2 = deepcopy(activation)
        self.activation3 = deepcopy(activation)
        self.activation4 = deepcopy(activation)
        self.activation5 = activation
        self._apply_initializer(initializer)
        self.softmax = nn.Softmax(dim=1)

        # Load model if path provided
        path_to_junior = customized_model_kwargs.get("model_path", None)
        if path_to_junior:
            self._params_copy(path=path_to_junior)

    def forward(self, input_dict, state=None, seq_lens=None):
        """
        Performs a forward pass through the model.

        This method processes the input observation through the layers and computes the action logits and value output.

        Args:
            input_dict (dict): A dictionary containing the observation data from RLlib.
            state: The state input (not used in this model).
            seq_lens: Sequence lengths (not used in this model).

        Returns:
            Tuple[torch.Tensor, None]: The action logits and the state (None in this case).
        """
        # Extract the observations from the input dictionary
        if isinstance(input_dict["obs"], np.ndarray):
            input = torch.from_numpy(input_dict["obs"])
        else:
            input = input_dict["obs"]

        x = input
        # Apply the layers and activations
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.activation3(self.layer3(x))
        x = self.activation4(self.layer4(x))

        # Action and value outputs
        action = self.act_layer(x)
        val_hidden = self.activation5(self.val_hidden_layer(x))
        value = self.val_layer(val_hidden)

        # Store value for value function calls later
        self._value_out = value.squeeze(-1)

        # Apply softmax to action
        action = self.softmax(action)

        # Return action logits and state
        return action, state

    def _extract_config(self, config) -> Tuple[List, any, any]:
        """
        Extracts model configuration parameters, including layer sizes, activation functions, and initializers.

        Args:
            config (dict): Configuration dictionary containing model parameters.

        Returns:
            Tuple[List, any, any]: A tuple containing:
                - A list of layer sizes.
                - The initializer for the layers.
                - The activation function to be used in the model.
        """
        layer_size = [1000, 1000, 1000, 1000]
        initializer = nn.init.orthogonal_
        activation = nn.ReLU()

        if not any([k in config.keys() for k in ["activation", "initializer", "layer1", "layer2", "layer3", "layer4"]]):
            warnings.warn("The custom dictionary did not have the correct keys. Using default model.")
            return layer_size, initializer, activation

        activation_option = {"leaky_relu": nn.LeakyReLU(inplace=True),
                             "relu": nn.ReLU(inplace=True),
                             "elu": nn.ELU(inplace=True)}
        initializer_option = {"O": nn.init.orthogonal_,
                              "RN": nn.init.normal_,
                              "RU": nn.init.uniform_,
                              "Z": nn.init.zeros_}

        if "activation" in config:
            activation = activation_option[config["activation"]]

        if "initializer" in config:
            initializer = initializer_option[config["initializer"]]

        if all([l in config for l in ["layer1", "layer2", "layer3", "layer4"]]):
            layer_size = [int(round(config[l])) for l in ["layer1", "layer2", "layer3", "layer4"]]

        return layer_size, initializer, activation

    def _apply_initializer(self, initializer):
        """
        Applies the specified initializer to the model layers.

        Args:
            initializer (Callable): The initializer function for the model weights.

        Returns:
            None.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initializer(m.weight)

    def load_my_state_dict(self, state_dict):
        """
        Loads the model's state dictionary, mapping the given state dictionary to the current model's parameters.

        This method adjusts the keys of the provided `state_dict` to match the current model's state dictionary
        and copies the weights from the provided dictionary into the corresponding parameters of the model.

        Args:
            state_dict (dict): The state dictionary containing the model weights. This can be a saved checkpoint
                from a pretrained model.

        Returns:
            None.
        """
        own_state = self.state_dict()

        for name, param in state_dict.items():
            name_corr = name
            if name.startswith("model."):
                name_corr = name[len("model."):]
            if name_corr not in own_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name_corr].copy_(param)

    def _params_copy(self, path: Path):
        """
         Copies the parameters from a pretrained model to the custom PyTorch model.

         This method loads the weights of a pretrained Junior or Senior model from the specified path
         and applies them to the custom PyTorch model.

         Args:
             path (Path): The file path to the saved model checkpoint.

         Returns:
             None.
         """
        checkpoint = torch.load(path)
        try:
            state_dict = checkpoint.get('state_dict', checkpoint)
        except AttributeError:
            print("Loading the Senior model")
            state_dict = checkpoint.state_dict()

        self.load_my_state_dict(state_dict)

    def value_function(self):
        """
        Returns the value function output of the model.

        This is used by RLlib to compute the value function during training.

        Returns:
            The value function output.
        """
        return self._value_out
