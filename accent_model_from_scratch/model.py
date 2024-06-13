from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, Bidirectional, GRU)

def accent_model(input_dim=13, units=200, recur_layers=2, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # First Bidir RNN and batchnorm layers
    bidir_GRU = Bidirectional(GRU(units, activation='relu',
        return_sequences=True))(input_data)
    bn_rnn = BatchNormalization()(bidir_GRU)
    
    # Subsequent RNN and batchnorm layers
    for _ in range(recur_layers-1):
        rnn_x = Bidirectional(GRU(units, activation='relu',
        return_sequences=True))(bn_rnn)
        bn_rnn = BatchNormalization()(rnn_x)
    
    # TimeDistributed Dense layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: x
    print(model.summary())
    return model
