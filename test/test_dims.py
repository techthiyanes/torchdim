import dim
from dim import Tensor, Dim, dims, stack, DimensionBindError, cat

from attn_ft import BertSelfAttention as BertSelfAttentionA
from attn_positional import BertSelfAttention as BertSelfAttentionB

from unittest import TestCase, main
import torch
import gc
import refcycle


from dim._C import _test_c, _n_levels_in_use

from contextlib import contextmanager
from time import perf_counter

@contextmanager
def measure(what):
    b = perf_counter()
    yield
    e = perf_counter()
    print (f"{what}: {e - b:.20f} seconds")

def triu(A):
   i,j = dims()
   a = A[i, j]
   zero = torch.tensor(0, dtype=torch.float) # XXX - torch.where is janky...
   return torch.where(i <= j, a, zero).positional(i, j)


class TestMin(TestCase):

    def setUp(self):
        gc.disable()
        gc.collect()

    def tearDown(self):
        nolevels = _n_levels_in_use() == 0
        if not nolevels:
            refcycle.garbage().export_image('garbage.png')
        gc.collect()
        assert nolevels, f"cleanup failed? {_n_levels_in_use()}"

    def test_manual_stuff(self):

        A_ = torch.rand(3, 4)
        B_ = torch.rand(4, 5)
        i,j,k = dims()
        A = A_[i,k]
        B = B_[k,j]
        C = (A.expand(j) * B.expand(i)).sum(k)
        self.assertTrue(torch.allclose(C.positional(i, j), torch.mm(A_, B_)))
        self.assertTrue(torch.allclose(torch.triu(A_, 0), triu(A_)))

        D_ = torch.randint(0, 3, (6,))
        d = dims()
        D = D_[d]

        A.gather([i], [D]).positional(k, d)

    def test_attn(self):

        batch_size = 1
        sequence_length = 4
        hidden_size = 6
        num_attention_heads = 3
        attention_probs_dropout_prob = 0.
        A = BertSelfAttentionA(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        B = BertSelfAttentionB(hidden_size, num_attention_heads, attention_probs_dropout_prob)

        A.load_state_dict(B.state_dict())
        hidden_state = torch.rand(batch_size, sequence_length, hidden_size)

        b_out = B(hidden_state)
        a_out = A(hidden_state)
        self.assertTrue(torch.allclose(a_out, b_out)) # why does a simple matmul not do the right thing?

        for approach in ('relative_key', 'relative_key_query'):
            num_attention_heads = 3
            hidden_size = 6
            A = BertSelfAttentionA(hidden_size, num_attention_heads, attention_probs_dropout_prob, approach, 4)
            B = BertSelfAttentionB(hidden_size, num_attention_heads, attention_probs_dropout_prob, approach, 4)
            A.load_state_dict(B.state_dict())

            hidden_state = torch.rand(2, 4, hidden_size)
            b_out = B(hidden_state)
            a_out = A(hidden_state)
            self.assertTrue(torch.allclose(a_out, b_out))

        num_attention_heads = 3
        hidden_size = 6
        A = BertSelfAttentionA(hidden_size, num_attention_heads, attention_probs_dropout_prob, None, 4)
        B = BertSelfAttentionB(hidden_size, num_attention_heads, attention_probs_dropout_prob, None, 4)
        A.load_state_dict(B.state_dict())

        hidden_state = torch.rand(2, 4, hidden_size)
        past_key_value = (torch.rand(2, num_attention_heads, 4, hidden_size//num_attention_heads),
                          torch.rand(2, num_attention_heads, 4, hidden_size//num_attention_heads))

        b_out = B(hidden_state, past_key_value=past_key_value)
        a_out = A(hidden_state, past_key_value=past_key_value)
        self.assertTrue(torch.allclose(a_out, b_out))

    def test_stack(self):
        i, j, d = dims()
        A = torch.rand(4, 5)
        r = stack([A[i,j]], d, j)
        #a, b = r.unbind(d)
        #self.assertTrue(torch.allclose(a.positional(i, j), i.expand(j).positional(i, j)))
        #self.assertTrue(torch.allclose(b.positional(i, j), j.expand(i).positional(i, j)))

    def test_max(self):
        ap = torch.rand(2, 3, 2)
        i, j, k = dims()
        a = ap[i, j, k]
        r, i0 = a.max(dim=k)
        self.assertTrue(torch.allclose(r.positional(i, j), ap.max(2)[0]))

    def test_mm(self):
        i, j, k, q = dims()
        a = torch.rand(3, 4)
        b = torch.rand(4, 5)
        a_ = a[i,k]
        b_ = b[k, j]
        q.size = 1
        r = (a_.expand(j, q)*b_.expand(i, q)).sum(k).positional(q, i, j)
        #r = (a_*b_).sum(k).positional(q, i, j)
        # print(r)
        # print(a @ b)

    def test_with_dims_split(self):
        a = torch.arange(3*12).view(3, 12)
        i, j, k = dims()
        k.size = 4
        r = a[i, (j, k)]
        x = r.positional(i, (j, k))
        self.assertTrue(torch.allclose(a, x))

    def test_hello(self):
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        i, j, k = dims()



        # r = A[i]*4
        r = (A[i, k] * B[k, j]).sum(k).positional(i, j)
        assert torch.allclose(r, A@B)

        assert A.sum() == A[i].sum((0, i))
        assert A.sum() == A[i].sum((-1, i))

        assert torch.allclose(A.sum(), A[i].sum(0, keepdim=True).sum( (0,i) ))
        assert torch.allclose(A[i].std(i, True), A.std(0, True))

        assert torch.allclose(A[i, k].max(i)[0].positional(k), A.max(0)[0])
        assert torch.allclose(A.sort(1)[0], A[i,k].sort(k)[0].positional(i,k))
        # XXX - chunk changes the size of a dimension, has to take a new dimension...
        # assert torch.allclose(A.chunk(2,1)[0], A[i, k].chunk(2, k)[0].positional(i, k))
        assert torch.allclose(A[i].renorm(1, i, 7).positional(i), A.renorm(1, 0, 7))
        kk = dims()
        # assert torch.allclose( torch.stack([A, A], 1), stack([A[i,k], A[i, k]], kk, k).positional(i, kk, k))

        k2 = dims()
        # r = cat((A[i, k], A[i,k]), k, k2)
        # assert torch.allclose(torch.cat([A, A], 1), r.positional(i, k2))
        # assert k2.size == 2*k.size

        assert torch.allclose(A.expand(5, -1, -1), A[i, k].expand(j).positional(j, i, k))
        z = dims()
        C = torch.arange(2)
        assert torch.allclose(A[:,0:2], A[i, k].gather(k, C[z]).positional(i, z))

        o,l = dims()
        o.size = 2
        r = A[i, k].reshape_dim(k, (o, l))
        assert torch.allclose(r.positional(i, o, l), A.view(-1, 2, 2))
        rr = r.reshape_dim((o, l), k)
        assert torch.allclose(A, rr.positional(i,k))

        r = i + k - 1
        r2 =  torch.arange(3)[:, None] + torch.arange(4)[None, :] - 1
        assert torch.allclose(r.positional(i, k), r2)

        # test with ...
        assert torch.allclose(A.T, A[..., k].positional(k))

        # test with dimlist
        a_,b_ = dims(lists=2)
        assert torch.allclose(A[i, a_].positional(*a_, i), A.T)
        # test with one bound dimlist
        assert torch.allclose(A[:, a_].positional(*a_), A.T)
        # test with a dimlist that will end up empty
        assert torch.allclose(A[i, b_, k].positional(i, k, *b_), A)
        # test with too few things
        print((A[i] + i))
        assert torch.allclose((A[i] + i).positional(i), A + torch.arange(3)[:, None])
        # test with too many elements
        try:
            A[1,...,1,1]
            raise NotImplemented()
        except IndexError:
            pass
        c, d = dims()
        c.size = 2
        assert torch.allclose(A[i, (c,d)].positional(i, c, d), A.view(3, 2, 2))

        assert torch.allclose(A[c + 1, c + 0].positional(c), A[torch.arange(2) + 1, torch.arange(2)])
        try:
            A[..., 3, ...]
            raise NotImplemented()
        except DimensionBindError:
            pass

        C = torch.rand(4, 7)
        c_, x, y, z = dims()

        a,b,c = C.split((3, 3, 1), dim=1)
        s = dims()
        ref = C.split((3,3,1), dim=1)
        t = C[s,c_].split((x,y,z), dim=c_)
        for a,b,d in zip(ref, t, (x,y,z)):
            assert torch.allclose(a, b.positional(s,d))

        D = torch.rand(3, 4, 5)
        assert torch.allclose(D.transpose(0, 1).flatten(1,2), D[i, k, j].positional((i, j)).positional(k))



        print(torch.rand_like(A[i, k]))
        print(torch.nn.functional.dropout(A[i, k]))

    def test_simple(self):
        i, j, k = dims()
        x = torch.rand(3, 4)
        z = x[i, j]
        print(z + z + z + z)
        print(z.positional(i, j))

    def test_mm_fuse(self):
        i, j, k = dims()
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)

        C = (A[i, k] * B[k, j]).sum(k).positional(i, j)
        assert torch.allclose(C, A @ B)

    def test_time_mm_fuse(self):
        i, j, k = dims()
        A = torch.rand(1, 1)
        B = torch.rand(1, 1)


        for _ in range(10):
            r0 = A + B
        for _ in range(10):
            a = A[i, j]
            b = B[i, j]
            r1 = a + b

        with measure('pp'):
            for _ in range(10000):
                A + B
        # magic_trace_stop_indicator()

        with measure('fc'):
            for _ in range(10000):

                a + b
        # while True:
        #         a + b
        # magic_trace_stop_indicator()


        assert torch.allclose(r1.positional(i,j), r0)

    def test_compare_dims(self):
        i, j = dims()
        i.size = 3
        j.size = 4
        print(i < j)

    def test_c(self):
        _test_c()

    def test_seg(self):
        A = torch.rand(3, 4)
        i, k = dims()
        i.size = 4
        k.size = 3
        r = i + k - 1

    def test_expand(self):
        A = torch.rand(3, 4)
        i = dims()
        assert list(A[i].expand(2, 4).positional(i).size()) == [3, 2, 4]


def do_stuff(a):
    i = dims()
    i.size = 4

if __name__ == '__main__':
    main()
