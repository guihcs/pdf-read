import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from tqdm.auto import tqdm
import math

def shift(t):
    pad = torch.zeros(t.shape[0], t.shape[1], 1).to(t.device)
    t = torch.cat([pad, t], dim=-1)
    t = t.view(t.shape[0], t.shape[2], t.shape[1])

    return t[:, 1:, :]


def pos_encode(max_len, d_model):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def rel_pos(pos_seq, demb):
    inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))

    sinusoid_inp = torch.ger(pos_seq, inv_freq)
    return torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)


class RelAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(RelAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qw = nn.Linear(d_model, d_model)
        self.kw = nn.Linear(d_model, d_model)
        self.vw = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(dropout)

        self.cw = nn.Linear(d_model, d_model)

        self.scale = 1 / (d_model ** 0.5)

    def forward(self, x, u, v, pos, mem=None, mask=None):

        if mem is None:
            mememb = x
        else:
            mememb = torch.cat([mem, x], dim=1)

        qw = self._reshape_mh(self.qw(mememb))
        kw = self._reshape_mh(self.kw(mememb))
        vw = self._reshape_mh(self.vw(mememb))

        rsp = self._reshape_mh(pos)
        ac = torch.einsum('ijk,imk->ijm', (qw + u), kw)

        bd = torch.einsum('ijk,ilk->ijl', (qw + v), rsp)
        bd = shift(bd)
        att = ac + bd
        att.mul_(self.scale)
        if mask is not None:
            att = att.masked_fill(mask != 0, -1e12)

        att = torch.softmax(att, dim=-1)
        att = self.drop(att)

        out = att @ vw.float()

        return self.cw(self._reshape_out(out))

    def _reshape_mh(self, x):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2], -1)

    def _reshape_out(self, x):
        x = x.reshape(-1, self.n_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.rmha = RelAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(0.1)

        self.ln2 = nn.LayerNorm(d_model)
        # self.ca = MHA(d_model, num_heads)
        self.ca = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)

        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dff, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x, y, u, v, pos, mem=None, mask1=None, mask2=None):
        nx = self.ln1(x)
        rl = self.dropout1(self.rmha(nx, u, v, pos, mem=mem, mask=mask1))
        av = nx + rl[:, -x.shape[1]:, :]
        no = self.ln2(av)

        atv = self.ca(no, y, y, attn_mask=mask2, need_weights=False)[0]
        fv = no + self.dropout2(atv)

        lv = self.ln3(fv)
        return lv + self.ff(lv)



class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, dff, num_layers):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff) for _ in range(num_layers)])
        self.u = nn.Parameter(torch.rand((1, 1, d_model // num_heads)))
        self.v = nn.Parameter(torch.rand((1, 1, d_model // num_heads)))

    def forward(self, x, y, mem=None, mask1=None, mask2=None):
        out = x
        new_mem = []
        ml = mem[0].shape[1] if mem is not None else 0
        pos_seq = torch.arange(x.shape[1] + ml - 1, -1, -1.0)
        pos = rel_pos(pos_seq, self.d_model).to(x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        for i, layer in enumerate(self.layers):
            out = layer(out, y, self.u, self.v, pos, mem=mem[i] if mem is not None else None, mask1=mask1, mask2=mask2)
            new_mem.append(out.detach())
        return out, new_mem


class VitXl(nn.Module):
    def __init__(self, vocab_size, vsize, d_model, num_heads, dff, num_layers, max_len=512, seq_len=1024, mem_len=512):
        super(VitXl, self).__init__()

        self.seq_len = seq_len
        self.mem_len = mem_len
        self.num_heads = num_heads

        self.v_encoder = nn.Sequential(
            nn.Linear(vsize, dff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dff, d_model),
            nn.Dropout(0.1)
        )

        self.pos = nn.Parameter(pos_encode(max_len, d_model), requires_grad=False)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.decoder = Decoder(d_model, num_heads, dff, num_layers)

    def forward(self, x, y):
        vx = self.vis_encode(x)
        return vx, self.text_decode(vx, y)[0]


    def vis_encode(self, x):
        x = self.v_encoder(x)
        x += self.pos[:x.shape[1], :].to(x.device)
        return self.encoder(x)


    def text_decode(self, vx, y, mem=None):
        ey = self.embedding(y)
        final_out = []
        for chunk in torch.split(ey, self.seq_len, dim=1):
            ml = mem[0].shape[1] if mem is not None else 0
            sl = chunk.shape[1] + ml
            m1 = torch.triu(torch.ones((1, sl, sl)), diagonal=1).to(chunk.device)
            m2 = torch.triu(torch.ones((y.shape[0], chunk.shape[1], vx.shape[1])), diagonal=1).to(chunk.device).repeat \
                (self.num_heads, 1, 1) * -1e12

            out, nm = checkpoint.checkpoint(self.decoder, chunk, vx, mem, m1, m2, use_reentrant=False)
            mem = [mv[:, -self.mem_len:, :] for mv in nm]
            final_out.append(out)

        return torch.cat(final_out, dim=1), mem



# vit = VitXl(tokenizer.vocab_size, 256, 384, 6, 1024, 4, max_len=800, seq_len=512, mem_len=64)
# vit.cuda(0)
# e1 = torch.rand((32, 800, 256))
# e2 = torch.randint(0, 3, (32, 6000))
#
# vx, ey = vit(e1.cuda(0), e2.cuda(0))
# print(vx.shape, ey.shape)




class PretrainVit(nn.Module):
    def __init__(self, vit, vocab_size, v_dim=256, n_dim=512, dim_feedforward=1024):
        super(PretrainVit, self).__init__()

        self.vit = vit
        self.vd = nn.Sequential(
            nn.Linear(n_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, v_dim),
            nn.Sigmoid()
        )
        self.td = nn.Sequential(
            nn.Linear(n_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, y):
        vx, vy = self.vit(x, y)
        return self.vd(vx), self.td(vy)


class PDFReader(nn.Module):
    def __init__(self, encoder, vocab_size, d_model=768, dim_feedforward=2048):
        super(PDFReader, self).__init__()
        self.encoder = encoder

        self.ll = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, y):
        # vx, vy = self.encoder(x, y)
        vx, vy = checkpoint.checkpoint(self.encoder, x, y, use_reentrant=False)

        return self.ll(vy)

    def read(self, tokenizer, x, ml=512, show_progress=False):

        with torch.no_grad():

            vx = self.encoder.vis_encode(x)

            sy = tokenizer.encode('<s>', add_special_tokens=False, return_tensors='pt').to(x.device).repeat(vx.shape[0],
                                                                                                            1)

            rg = range(ml - sy.shape[1])

            if show_progress:
                rg = tqdm(rg)

            mem = None

            yp = sy

            for i in rg:
                td, nm = self.encoder.text_decode(vx, yp, mem=mem)
                et = td[:, -1, :]
                dc = self.ll(et).exp().argmax(dim=-1).unsqueeze(1)
                sy = torch.cat([sy, dc], dim=-1)

                yp = torch.cat([yp, dc], dim=-1)

                if yp.shape[1] >= self.encoder.seq_len:
                    mem = nm
                    yp = dc

            return sy
