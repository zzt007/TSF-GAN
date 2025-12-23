import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA_LAUNCH_BLOCKING=1



def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).to(device),
            torch.zeros(self.num_layers, batch, self.h_dim).to(device)
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        device = obs_traj.device

        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))

        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )

        state_tuple = self.init_hidden(batch, device)

        output, state = self.encoder(obs_traj_embedding, state_tuple)

        final_h = state[0]

        return final_h

class TrafficEncoder(nn.Module):
    def __init__(self, traffic_state_dim=5,embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(TrafficEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.traffic_embedding = nn.Embedding(traffic_state_dim, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).to(device),
            torch.zeros(self.num_layers, batch, self.h_dim).to(device)
        )

    def forward(self, traffic_state):
        traffic_state = traffic_state.long()

        batch_size = traffic_state.size(1)
        seq_len = traffic_state.size(0)
        device = traffic_state.device

        max_index = torch.max(traffic_state).item() 
        embedding = torch.nn.Embedding(max_index + 1, self.embedding_dim).to(device) 
        traffic_state_embedding = embedding(traffic_state) 
        
        traffic_state_embedding = torch.squeeze(traffic_state_embedding, -2)
        state_tuple = self.init_hidden(batch_size, device)
        output, state = self.encoder(traffic_state_embedding, state_tuple)

        final_h = state[0]
        return final_h

class SizeEncoder(nn.Module):
    def __init__(self, agent_type_dim=6, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(SizeEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.size_layer = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).to(device),
            torch.zeros(self.num_layers, batch, self.h_dim).to(device)
        )

    def forward(self, size):
        batch_size = size.size(1)
        device = size.device

        size_embedding = self.size_layer(size)

        state_tuple = self.init_hidden(batch_size, device)

        output, state = self.encoder(size_embedding, state_tuple)

        final_h = state[0]

        return final_h

class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(StateEncoder, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.state_layer = nn.Linear(4, embedding_dim)  # vx, vy, ax, ay
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).to(device),
            torch.zeros(self.num_layers, batch, self.h_dim).to(device)
        )

    def forward(self, state):
        batch = state.size(1)
        device = state.device
        state_embedding = self.state_layer(state)
        state_tuple = self.init_hidden(batch, device)
        output, state = self.encoder(state_embedding, state_tuple)
        final_h = state[0]
        return final_h

class SpectralEncoder(nn.Module):
    def __init__(self,
                 embedding_dim=64,
                 h_dim=64,
                 mlp_dim=1024,
                 num_layers=1,
                 dropout=0.0):
        super(SpectralEncoder, self).__init__()
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.dft_embedding = nn.Linear(4, embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).to(self.device),
            torch.zeros(self.num_layers, batch, self.h_dim).to(self.device))
        
    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        
        obs_traj_cpu = obs_traj.cpu()
        x_fft_cpu = torch.fft.fft(obs_traj_cpu[:, :, 0],dim=0,norm='ortho')
        y_fft_cpu = torch.fft.fft(obs_traj_cpu[:, :, 1],dim=0,norm='ortho')

        x_fft = x_fft_cpu.to('cuda')
        y_fft = y_fft_cpu.to('cuda')

        x_fft_real = torch.view_as_real(x_fft)
        y_fft_real = torch.view_as_real(y_fft)

        fft_result = torch.cat([x_fft_real, y_fft_real],dim=-1)
        
        fft_embedding = self.dft_embedding(fft_result.reshape(-1, 4))
        fft_embedding = fft_embedding.view(
            -1, batch, self.embedding_dim)
        

        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(fft_embedding, state_tuple)
        final_h = state[0]

        return final_h

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='atten_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'atten_net':
                self.pool_net = AttenPoolNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )


            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, vx,vy, agent_type):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2) #前一个位置
        - last_pos_rel: Tensor of shape (batch, 2) #相对位置
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim) #隐藏状态和单元状态
        - seq_start_end: A list of tuples which delimit sequences within batch #序列开始和节俗的索引
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)

            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))

            curr_pos = rel_pos + last_pos


            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos, vx, vy, agent_type)

                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])


            embedding_input = rel_pos


            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)

            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """

        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h



class AttenPoolNet(PoolHiddenNet):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(AttenPoolNet, self).__init__(embedding_dim, h_dim, mlp_dim, bottleneck_dim,
                                           activation, batch_norm, dropout)

        self.velocity_embedding = nn.Linear(2, embedding_dim)
        self.agent_type_embedding = nn.Linear(6, embedding_dim)
        self.attention_mlp = make_mlp(
            [embedding_dim, mlp_dim, 1],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def compute_attention_weights(self, rel_pos_embedding, velocity_embedding, agent_type_embedding):
        hadamard_product = rel_pos_embedding * velocity_embedding * agent_type_embedding
        attention_scores = self.attention_mlp(hadamard_product)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights

    def forward(self, h_states, seq_start_end, end_pos, vx, vy, agent_type):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            
            curr_agent_type_from_dataset = agent_type[0,:,:]
            curr_agent_type = curr_agent_type_from_dataset[start:end]

            curr_hidden_repeated = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_repeated = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_transposed = curr_end_pos.repeat(1, num_ped).view(num_ped * num_ped, -1)
            curr_rel_pos = curr_end_pos_repeated - curr_end_pos_transposed
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)


            curr_vx = vx[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_vy = vy[-1, start:end].repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_velocity = torch.cat((curr_vx, curr_vy), dim=1)
            curr_velocity_embedding = self.velocity_embedding(curr_velocity)
            
            curr_agent_type_repeated = curr_agent_type.repeat_interleave(num_ped).view(num_ped * num_ped, -1)
            curr_agent_type_repeated = curr_agent_type.repeat(num_ped, 6)
            curr_agent_type_embedding = self.agent_type_embedding(curr_agent_type_repeated)

            attention_weights = self.compute_attention_weights(curr_rel_embedding, 
                                                               curr_velocity_embedding,
                                                               curr_agent_type_embedding)


            weighted_h_input = torch.cat([curr_rel_embedding, curr_hidden_repeated], dim=1)
            weighted_h_input *= 0.05 * attention_weights.view(-1, 1)

            # MLP processing as in PoolHiddenNet
            curr_pool_h = self.mlp_pre_pool(weighted_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]

            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
        traffic_h_dim = 64
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None
        self.traffic_h_dim=traffic_h_dim

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=64,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.traffic_encoder = TrafficEncoder(traffic_state_dim=5,embedding_dim=64, h_dim=traffic_h_dim)
        self.vehicle_encoder = VehicleEncoder(agent_type_dim=6, embedding_dim=64, h_dim=64)
        self.state_encoder = StateEncoder(embedding_dim=embedding_dim, h_dim=64)
        self.spectral_encoder = SpectralEncoder(embedding_dim=embedding_dim, h_dim=64)


        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'atten_net':
            self.pool_net = AttenPoolNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                activation=activation,
                bottleneck_dim=bottleneck_dim,
                batch_norm=batch_norm,
                dropout=dropout
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim*4 + bottleneck_dim + traffic_h_dim

        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end,  vx, vy, ax, ay, agent_type, size, traffic_state, user_noise=None,):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        vehicle_encoding = self.vehicle_encoder(agent_type, size)
        state_encoding = self.state_encoder(torch.cat([vx, vy, ax, ay], dim=2))
        traffic_encoding = self.traffic_encoder(traffic_state)
        spectral_encoding = self.spectral_encoder(obs_traj_rel)

        combined_encoding = torch.cat([final_encoder_h, vehicle_encoding, state_encoding, spectral_encoding], dim=2)


        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]

            pool_h = self.pool_net(combined_encoding, seq_start_end, end_pos,vx,vy, agent_type)



            mlp_decoder_context_input = torch.cat(
                [combined_encoding.view(-1, self.encoder_h_dim*4), 
                 pool_h, 
                 traffic_encoding.view(-1, self.traffic_h_dim)], 
                dim=1)



        else:
            mlp_decoder_context_input = combined_encoding.view(
                -1, self.encoder_h_dim)


        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input

        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
            vx,vy,
            agent_type
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)

        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores
