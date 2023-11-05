"""
Multi-gate Mixture-of-Experts model implementation.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng and modified by the Anonymous authors
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec, Dense
from tensorflow.keras.models import Model
class MMoE(Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                #  num_experts,
                 experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation=None,
                 gate_bias_initializer='zeros',
                 gate_bias_regularizer=None,
                 gate_bias_constraint=None,
                 gate_kernel_initializer='VarianceScaling',
                 gate_kernel_regularizer=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 dropout_rate=0.0,
                 tao=None,
                 epochs=60,
                 **kwargs):
        """
         Method for instantiating MMoE layer.

        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param gate_activation: Activation function of the gate weights
        :param gate_bias_initializer: Initializer for the gate bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param gate_kernel_initializer: Initializer for the gate weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        if tao is not None:
            self.tao=tao
        else:
            self.tao = 1/num_tasks
        self.epochs = epochs
        # Hidden nodes parameter
        self.units = units
        self.num_experts = len(experts)
        self.num_tasks = num_tasks

        # self.experts suppose to be a list
        self.experts = experts
        self.gate_kernels = None
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_gate_bias = use_gate_bias
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Keras parameter
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        super(MMoE, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Method for creating the layer weights.

        :param input_shape: Keras tensor (future input to layer)
                            or list/tuple of Keras tensors to reference
                            for weight shape computations
        """
        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = input_shape[-1]
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_experts),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        ) for i in range(self.num_tasks)]

        # Initialize gate bias (number of experts * number of tasks)
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias_task_{}'.format(i),
                shape=(self.num_experts,),
                initializer=self.gate_bias_initializer,
                regularizer=self.gate_bias_regularizer,
                constraint=self.gate_bias_constraint
            ) for i in range(self.num_tasks)]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dimension})

        super(MMoE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.

        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        gate_outputs = []
        final_outputs = []
        expert_outputs_list = []
        for exp in self.experts:
            expert_outputs_list.append(exp(inputs))
        expert_outputs = tf.convert_to_tensor(expert_outputs_list)
        expert_outputs = tf.transpose(expert_outputs,(1,2,0))

        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = K.dot(x=inputs, y=gate_kernel)
            if self.use_gate_bias:
                gate_output = K.bias_add(x=gate_output, bias=self.gate_bias[index])
            gate_output = self.gate_activation[index](gate_output)
            gate_outputs.append(gate_output)
    

        for gate_output in gate_outputs:
            expanded_gate_output = K.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(K.sum(weighted_expert_output, axis=2))
        return final_outputs

    def compute_output_shape(self, input_shape):
        """
        Method for computing the output shape of the MMoE layer.

        :param input_shape: Shape tuple (tuple of integers)
        :return: List of input shape tuple where the size of the list is equal to the number of tasks
        """
        assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

