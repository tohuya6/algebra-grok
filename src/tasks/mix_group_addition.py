import random
import torch
import string
import numpy
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics import CyclicGroup, DihedralGroup
from .magma import CyclicMonoid

class MixGroupAddition:
    """
    A generator for long sequences of group addition problems
    where one or more groups are randomly chosen for the sequence,
    and the assignment of group elements to variables is also
    selected to be a fixed random map used for the whole sequence.
    A single sequence can include facts from more than one group.
    Intended to mimic an in-context learning scenario.
    """
    def __init__(self, num_symbols: int = 16, max_order: int = 10,
            mix: float = 0.5, holdout_zero: bool = False, seed: int = 42,
            min_order: int = None) -> None:
        assert(max_order <= num_symbols)
        assert(min_order is None or min_order <= max_order), \
            f"min_order ({min_order}) must be <= max_order ({max_order})"
        self.task_name = self._task_name()
        self.num_symbols = num_symbols
        self.max_order = max_order
        # Lower bound on a sampled group's order; None -> each subclass's default
        # minimum. Setting min_order == max_order pins the order (with mix=0 that
        # gives one constant group every run -> clean fixed_p sweep).
        self.min_order = min_order
        self.mix = mix
        self.holdout_zero = holdout_zero
        self.seed = seed
        self.prng = random.Random(seed)

        # Setup the vocabulary
        variable_symbols = string.digits + string.ascii_letters
        self.vocab = [variable_symbols[i] for i in range(num_symbols)]
        self.predict_token_id = len(self.vocab)
        self.vocab.append('=')
        self.start_token_id = len(self.vocab)
        self.vocab.append('^')
        self.pad_token_id = self.start_token_id
        self.sep_token_id = len(self.vocab)
        self.vocab.append(',')
        self.vocab_size = len(self.vocab)
        self.numfor = { v: i for i, v in enumerate(self.vocab) }

        # Frozen random permutation of element-slots used by fixed_p to decide
        # *which* slots are pinned (see _fixed_slots). Seeded from self.seed so
        # it's reproducible; uses its own RNG so the self.prng draw stream (and
        # thus fixed_p=0 behaviour) stays byte-for-byte unchanged.
        self.fixed_perm = random.Random(seed).sample(range(num_symbols), num_symbols)

    def _task_name(self):
        return 'mixgroup'
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(num_symbols={self.num_symbols}, "
                f"max_order={self.max_order}, min_order={self.min_order}, mix={self.mix}, "
                f"holdout_zero={self.holdout_zero}, seed={self.seed})")

    def _fixed_slots(self, total_order: int, fixed_p: float) -> set:
        '''
        The element-slots (0..total_order-1) pinned to their canonical token by
        fixed_p. Selection order is a frozen random permutation (self.fixed_perm)
        rather than the low-index prefix, so the pinned elements aren't biased
        toward the identity/low-order slots. The set is stable across runs (hence
        memorizable) and nested as fixed_p grows: fixed_p=0 -> empty set,
        fixed_p=1 -> every slot.
        '''
        n_fixed = round(fixed_p * total_order)
        slots_by_pi = [s for s in self.fixed_perm if s < total_order]
        return set(slots_by_pi[:n_fixed])

    def sample_batch(self, batch_size: int,
            k_shots: int = 200, hold_out: int | list = 0,
            commute_out: bool = True,
            max_length: int = 1024,
            fixed_groups: list = None,
            fixed_p: float = 0.0):
        '''
        Returns a batch of batch_size examples as tensors, each demonstrating
        a run of k_shots of group addition facts drawn from a set of groups.

        If hold_out is provided, the given number of facts is
        held out of each run (always holding out commutative inverses if
        commute_out is set), and each run ends with a held out sample.
        If hold_out is set to a number, then the specified number of
        held out facts will be randomly chosen. If hold_out is a
        list of pairs of integers, then the corresponding entries in the
        cayley tables will be held out, and the first entry of the list
        will be used as the final test question.

        If fixed_groups is provided, the groups are from the list given;
        otherwise the groups in each run are chosen randomly.

        fixed_p partially pins the vocabulary: round(fixed_p * order) element-slots
        take their canonical token self.vocab[i] (consistent across runs, hence
        memorizable/parametric), while the rest draw a fresh random token each run
        (symbolic). Which slots are pinned is chosen by a frozen random permutation
        (see _fixed_slots), so the pinned elements aren't biased toward the
        low-index/identity slots. fixed_p=0 is the all-variable regime; fixed_p=1
        pins every slot to the canonical vocabulary self.vocab[:order].

        The return structure provides lists of groups, their orders (sizes),
        the vocabulary for each run, the subset of that vocabulary held fixed by
        fixed_p, and a mask showing where all the predictive tokens ("=" signs) are.

        Calls sample_run to do the work of sampling individual sequences.
        '''

        # Enforce max_length. Every shot emits exactly 5 tokens (',a b = c'), so the run
        # is 5*k_shots tokens and the model sees 5*k_shots-1 input tokens (inputs = [:, :-1]).
        # Fail loudly here rather than let the sequence overflow the model's block_size
        # downstream (which surfaces as a cryptic positional-embedding index error).
        n_input_tokens = 5 * k_shots - 1
        if n_input_tokens > max_length:
            raise ValueError(
                f"k_shots={k_shots} produces {n_input_tokens} input tokens, over "
                f"max_length={max_length}. Use k_shots <= {(max_length + 1) // 5} "
                f"or a larger block_size.")

        expressions, g, o, v = zip(*[
            self.sample_run(k_shots, hold_out, commute_out, fixed_groups, fixed_p)
                for _ in range(batch_size)])
        tensor = self.tensor_from_expression(expressions)

        # Every token is a goal
        return {
            "inputs": tensor[:,:-1],
            "targets": tensor[:,1:],
            "groups": g,
            "orders": o,
            "vocabulary": [''.join(voc) for voc in v],
            # Tokens pinned to their canonical value by fixed_p (the slots chosen
            # by self._fixed_slots, in a frozen random order); '' when fixed_p == 0.
            "fixed_vocabulary": [''.join(voc[s] for s in sorted(self._fixed_slots(len(voc), fixed_p)))
                                 for voc in v],
            "prediction_mask": (tensor[:,:-1] == self.predict_token_id)
        }

    def tensor_from_expression(self, expressions):
        '''
        Convert expression strings to tensor of token IDs.
        
        Takes nested structures of strings (characters representing tokens) and
        recursively converts them to their corresponding integer token IDs using
        the vocabulary mapping, then returns as a PyTorch tensor.
        
        Args:
            expressions: String or nested list/tuple of strings, where each 
                        character is a vocabulary token to be converted to its ID
        
        Returns:
            torch.LongTensor: Tensor of token IDs with the same nesting structure
                            as the input
        '''
        def recursive_numfor(e):
            if isinstance(e, (list, tuple)):
                return [recursive_numfor(el) for el in e]
            return [self.numfor[el] for el in e]
        return torch.tensor(recursive_numfor(expressions), dtype=torch.long)

    def sample_run(self, k_shots: int, hold_out: int | list = 0, commute_out: bool = True,
            fixed_groups = None, fixed_p: float = 0.0):
        '''
        Returns a single randomly-generated sequence of group facts as a string,
        chosing a random list of groups to use and a random vocabulary.
        
        Returns the sequence as a string, along with a list of the groups used,
        sizes of the groups, and the vocablary selected (with the first letter
        in the vocabulary string corresponding to the first element of
        the first group, and so on).

        The run can be controlled with the arguments; their meaning is the
        same as described in sample_batch.

        Calls sample_facts to do the work of generating a string that
        demonstrates facts from groups chosen by this method.
        '''

        # Sample random groups 
        if fixed_groups is not None:
            Glist = fixed_groups
        else:
            Glist = self.sample_groups()
        orders = [G.order() for G in Glist]
        total_order = sum(orders)
        assert (fixed_groups is not None or
                (max(orders) <= self.max_order and total_order <= self.num_symbols) )

        # Select a vocabulary: the slots chosen by fixed_p (self._fixed_slots,
        # picked in a frozen random order) take their canonical token self.vocab[i]
        # -- same across runs, hence memorizable/parametric; the rest draw a fresh
        # random token from the remaining symbols (-> symbolic). fixed_p=0 is the
        # all-variable case; fixed_p=1 pins every slot to self.vocab[:total_order].
        elems = [[(g, i) for g in G.generate()] for i, G in enumerate(Glist)]
        all_elems = sum(elems, [])
        fixed_slots = self._fixed_slots(total_order, fixed_p)
        n_fixed = len(fixed_slots)
        pool = [self.vocab[j] for j in range(self.num_symbols) if j not in fixed_slots]
        while True:
            rand_words = (self.prng.sample(pool, total_order - n_fixed)
                          if n_fixed < total_order else [])
            it = iter(rand_words)
            vocab = [self.vocab[i] if i in fixed_slots else next(it)
                     for i in range(total_order)]
            wordfor = { g: vocab[i] for i, g in enumerate(all_elems) }
            # holdout_zero forbids a *variable* identity (a slot not in fixed_slots)
            # from being '0'; fixed slots keep their canonical token regardless.
            offset, collide = 0, False
            for E in elems:
                if offset not in fixed_slots and wordfor[E[0]] == '0':
                    collide = True
                    break
                offset += len(E)
            if not self.holdout_zero or not collide:
                break

        # Create list of all possible facts from all groups' Cayley tables
        facts = [(a, b) for E in elems for a in E for b in E]

        # Hold out some facts
        held_out = []
        if isinstance(hold_out, int):
            while len(held_out) < hold_out:
                (a, b) = facts.pop(self.prng.randrange(0, len(facts)))
                held_out.append((a, b))
                if commute_out:
                    if (b, a) in facts:
                        facts.remove((b, a))
                        held_out.append((b, a))
        elif isinstance(hold_out, list):
            for (ai, bi) in hold_out:
                a, b = (all_elems[ai], all_elems[bi])
                if (a, b) in facts:
                    held_out.append((a, b))
                    facts.remove((a, b))
                if commute_out:
                    if (b, a) in facts:
                        facts.remove((b, a))
                        held_out.append((b, a))

        return self.sample_facts(k_shots, wordfor, facts, held_out), Glist, orders, vocab

    def sample_groups(self):
        '''
        Sample groups randomly according to the parameters of this task class,
        adding additional groups with p(self.mix), stopping if all available
        symbols have been used.
        '''
        total_order = 0
        Glist = []
        while True:
            G = self._sample_group(self.num_symbols - total_order)
            if G is None:
                break
            Glist.append(G)
            total_order += G.order()
            if self.prng.random() > self.mix:
                break
        return Glist

    def sample_facts(self, k_shots: int, wordfor: dict, facts: list, held_out: list):
        '''
        Generates a string demonstrating random sample of group facts.

        Each group element is represented as (g, n) where g is a sympy
        permutation group element, and n is a number used to disambiguate
        different instances of groups that should have their own vocabulary.

        Given a list of facts of the form [((a, n), (b, n)), ...] and a wordfor
        dictionary that maps each element pairs (a, n) -> 'a' to a vocabulary
        character, generates a string of random fact statements, by randomly
        sampling facts from the list and then computing (c, n) = (a * b, n)
        and then looking up (a, n), (b, n), and (c, n) in the wordfor dictionary
        and producing the output pattern ",ab=c,de=f,gh=i,jk=l"...

        If a set of facts is included in the held_out list, then the sequence
        will end with the first fact in the list.
        '''
        sequence = []
        # Randomly sample from the fact table, excepting held-out facts until the end
        for _ in range(k_shots - (1 if held_out else 0)):
            a, b = self.prng.choice(facts)
            c = (a[0] * b[0], a[1])
            sequence.extend([',', wordfor[a], wordfor[b], '=', wordfor[c]])
        # If holding out some facts, end with a held-out fact
        if held_out:
            a, b = held_out[0]
            c = (a[0] * b[0], a[1])
            sequence.extend([',', wordfor[a], wordfor[b], '=', wordfor[c]])
        return ''.join(sequence)

    def stringify(self, seq):
        '''
        For debugging and unit tests: given an instance of the return value
        of sample_batch (or various fields), make a readable string
        describing the instance.
        '''
        if isinstance(seq, dict):
            return ''.join([f'\n{k}:\n{self.stringify(v)}' for k, v in seq.items()])
        if isinstance(seq, str):
            return f'"{seq}"'
        if isinstance(seq, PermutationGroup):
            if seq.is_cyclic:
                return f'CyclicGroup({seq.order()})'
            elif seq.is_dihedral:
                return f'DihedralGroup({seq.order() // 2})'
            return str(seq)
        if isinstance(seq, int):
            return str(seq)
        if isinstance(seq, (list, tuple)) and len(seq) and (
                isinstance(seq[0], (PermutationGroup, int))):
            return ' '.join([self.stringify(i) for i in seq])
        if isinstance(seq, (list, tuple)):
            return '\n'.join([self.stringify(i) for i in seq])
        if numpy.ndim(seq) > 1:
            return '\n'.join([self.stringify(d) for d in seq])
        if numpy.ndim(seq) == 1:
            return ''.join([
                self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else '%'
                for i in seq])
        return str(seq)

    def summarize(self, batches, predictions, accuracy, length=72):
        '''
        For logging: summarize predictions within the first run of a batch.
        '''
        batch = batches[0]
        predictions = predictions[0]
        def charfor(a):
            return self.vocab[int(a)] if 0 <= int(a) < len(self.vocab) else '%'
        summary = ''
        # 12 samples of the test cases
        for index in range(min(12, len(batch['inputs']))):
            inputs = batch['inputs'][index][-length:]
            targets = batch['targets'][index][-length:]
            pred = predictions[index][-length:]
            # include lines of raw output
            # summary += ''.join(['\n' + self.stringify(d) for d in [inputs, targets, pred]])
            summary_chars = []
            for i, a in enumerate(inputs):
                summary_chars.append(charfor(a))
                if int(a) == self.predict_token_id:
                    summary_chars.append(f'[{charfor(pred[i])}]')
            summary += '\n' + (''.join(summary_chars) + charfor(targets[-1]))[-length:]
        return summary

class MixCyclicGroupAddition(MixGroupAddition):
    """
    Uniformly sample cyclic groups of order at least 3, up to the maximum order
    """
    def _task_name(self):
        return 'mixcyclic'

    def _sample_group(self, max_order: int = None):
        hi = min(o for o in [self.max_order, max_order] if o is not None)
        lo = self.min_order if self.min_order is not None else 3   # cyclic order == modulus
        if hi < lo or hi < 3:
            return None
        modulus = self.prng.randrange(lo, hi + 1)
        return CyclicGroup(modulus)

    def _all_groups(self):
        return [CyclicGroup(i) for i in range(3, self.max_order + 1)]

class MixDihedralGroupAddition(MixGroupAddition):
    """
    Uniformly sample dihedral groups of order at least 4, up to the maximum order
    """
    def _task_name(self):
        return 'mixdihedral'

    def _sample_group(self, max_order: int = None):
        hi = min(o for o in [self.max_order, max_order] if o is not None)
        # DihedralGroup(modulus) has order 2*modulus; min_order is a bound on order.
        lo_mod = max(2, (self.min_order + 1) // 2) if self.min_order is not None else 2
        if hi // 2 < lo_mod:
            return None
        modulus = self.prng.randrange(lo_mod, hi // 2 + 1)
        return DihedralGroup(modulus)

    def _all_groups(self):
        return [DihedralGroup(i) for i in range(2, self.max_order // 2 + 1)]

class MixRosetteGroupAddition(MixGroupAddition):
    """
    Uniformly sample cyclic or dihedral groups of order at least 3, up to the maximum order
    """
    def _task_name(self):
        return 'mixrosette'

    def _sample_group(self, max_order: int = None):
        hi = min(o for o in [self.max_order, max_order] if o is not None)
        # min_order bounds the group order: cyclic order == index, dihedral order == 2*modulus.
        lo_c = max(3, self.min_order) if self.min_order is not None else 3
        lo_d = max(2, (self.min_order + 1) // 2) if self.min_order is not None else 2
        num_cyclic = max(0, hi - lo_c + 1)
        num_dihedral = max(0, hi // 2 - lo_d + 1)
        if num_cyclic + num_dihedral < 1:
            return None
        which_group = self.prng.randrange(num_cyclic + num_dihedral)
        if which_group < num_cyclic:
            return CyclicGroup(which_group + lo_c)
        else:
            return DihedralGroup((which_group - num_cyclic) + lo_d)
    
    def _all_groups(self):
        return [CyclicGroup(i) for i in range(3, self.max_order + 1)] + [DihedralGroup(i) for i in range(2, self.max_order // 2 + 1)]

class MixMonoidAddition(MixGroupAddition):
    """
    Sample cyclic or dihedral groups or cyclic monoids.
    """
    def _task_name(self):
        return 'mixmonoid'

    def _sample_group(self, max_order: int = None):
        hi = min(o for o in [self.max_order, max_order] if o is not None)
        # min_order bounds the order: cyclic/monoid order == index, dihedral == 2*modulus.
        lo_c = max(2, self.min_order) if self.min_order is not None else 2
        lo_d = max(2, (self.min_order + 1) // 2) if self.min_order is not None else 2
        num_cyclic = max(0, hi - lo_c + 1)
        num_dihedral = max(0, hi // 2 - lo_d + 1)
        if num_cyclic + num_dihedral < 1:
            return None
        if num_dihedral > 0 and self.prng.randrange(2) == 0:
            modulus = self.prng.randrange(lo_d, hi // 2 + 1)
            return DihedralGroup(modulus)
        modulus = self.prng.randrange(lo_c, hi + 1)
        if self.prng.randrange(2) == 0:
            return CyclicGroup(modulus)
        modulus -= 1
        order = self.prng.randrange(modulus + 1, hi + 1)
        return CyclicMonoid(order, modulus)

def _unit_test():
    import re
    def eqstring(a, b):
        # Remove space at start of lines
        [a, _], [b, _] = [re.subn(r'((?<=\n)|^) *', '', s) for s in [a, b]]
        if len(a) != len(b):
            print(f'Difference in length {len(a)} vs {len(b)}')
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                print(f'Difference at index: {i}: "{a[i:i+3]}" vs "{b[i:i+3]}"')
                print(a)
                break
        return a == b
    a = MixRosetteGroupAddition(max_order=6, num_symbols=12, holdout_zero=True)
    batch = a.sample_batch(batch_size=3, k_shots=12, fixed_p=1.0, hold_out=1)
    # Remove prediction_mask from batch before stringifying
    batch_without_mask = {k: v for k, v in batch.items() if k != 'prediction_mask'}
    assert eqstring(
        a.stringify(a.sample_batch(batch_size=3, k_shots=12, fixed_p=1.0, hold_out=1)), '''
        inputs:
        ,10=1,31=4,41=5,21=3,05=5,02=2,23=5,31=4,10=1,23=5,11=2,40=
        ,87=9,7b=6,44=2,66=6,13=4,35=2,ba=8,53=2,34=1,a7=8,88=6,32=
        ,44=3,33=1,13=4,43=2,21=3,01=1,13=4,01=1,21=3,23=0,14=0,40=
        targets:
        10=1,31=4,41=5,21=3,05=5,02=2,23=5,31=4,10=1,23=5,11=2,40=4
        87=9,7b=6,44=2,66=6,13=4,35=2,ba=8,53=2,34=1,a7=8,88=6,32=5
        44=3,33=1,13=4,43=2,21=3,01=1,13=4,01=1,21=3,23=0,14=0,40=4
        groups:
        CyclicGroup(6)
        CyclicGroup(6) DihedralGroup(3)
        CyclicGroup(5)
        orders:
        6
        6 6
        5
        vocabulary:
        "012345"
        "0123456789ab"
        "01234"
        fixed_vocabulary:
        "012345"
        "0123456789ab"
        "01234"
        prediction_mask:
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001''')
    # Skip forward for a case where holdout_zero makes a difference
    a.stringify(a.sample_batch(batch_size=3, k_shots=12))
    assert eqstring(
        a.stringify(a.sample_batch(batch_size=3, k_shots=12)), '''
        inputs:
        ,88=8,85=5,0b=5,b7=8,b0=5,8b=b,b0=5,55=b,80=0,80=0,75=0,58=
        ,a2=8,aa=2,8a=a,22=a,aa=2,a8=a,28=2,aa=2,28=2,22=a,a8=a,2a=
        ,71=1,22=a,33=a,33=a,77=7,7a=a,79=9,33=a,7a=a,73=3,99=7,7a=
        targets:
        88=8,85=5,0b=5,b7=8,b0=5,8b=b,b0=5,55=b,80=0,80=0,75=0,58=5
        a2=8,aa=2,8a=a,22=a,aa=2,a8=a,28=2,aa=2,28=2,22=a,a8=a,2a=8
        71=1,22=a,33=a,33=a,77=7,7a=a,79=9,33=a,7a=a,73=3,99=7,7a=a
        groups:
        CyclicGroup(5)
        CyclicGroup(3)
        CyclicGroup(6)
        orders:
        5
        3
        6
        vocabulary:
        "8b057"
        "82a"
        "7139a2"
        fixed_vocabulary:
        ""
        ""
        ""
        prediction_mask:
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001''')
    a = MixDihedralGroupAddition(max_order=14)
    assert eqstring(
        a.stringify(a.sample_batch(batch_size=3, k_shots=12)), '''
        inputs:
        ,02=1,58=5,8f=f,72=f,26=1,43=8,59=e,95=e,04=6,a0=e,b8=b,6e=
        ,00=5,d2=3,6c=2,15=1,b1=c,d0=4,3f=0,e8=7,c1=b,ae=e,4b=2,b2=
        ,93=3,39=3,d9=d,3d=1,d3=1,11=9,d3=1,d3=1,1d=3,d9=d,9d=d,11=
        targets:
        02=1,58=5,8f=f,72=f,26=1,43=8,59=e,95=e,04=6,a0=e,b8=b,6e=7
        00=5,d2=3,6c=2,15=1,b1=c,d0=4,3f=0,e8=7,c1=b,ae=e,4b=2,b2=4
        93=3,39=3,d9=d,3d=1,d3=1,11=9,d3=1,d3=1,1d=3,d9=d,9d=d,11=9
        groups:
        DihedralGroup(7)
        DihedralGroup(6) DihedralGroup(2)
        DihedralGroup(2)
        orders:
        14
        12 4
        4
        vocabulary:
        "83e2b1fa607594"
        "5b6f423c109da78e"
        "91d3"
        fixed_vocabulary:
        ""
        ""
        ""
        prediction_mask:
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001''')
    a = MixCyclicGroupAddition(max_order=13)
    assert eqstring(
        a.stringify(a.sample_batch(batch_size=3, k_shots=12, fixed_p=1.0)), '''
        inputs:
        ,29=b,20=2,ee=f,a9=6,19=a,b8=6,84=c,08=8,07=7,1a=b,43=7,47=
        ,65=0,23=5,83=0,76=2,81=9,63=9,49=2,26=8,52=7,69=4,32=5,94=
        ,02=2,20=2,12=0,11=2,02=2,10=1,12=0,01=1,01=1,20=2,01=1,12=
        targets:
        29=b,20=2,ee=f,a9=6,19=a,b8=6,84=c,08=8,07=7,1a=b,43=7,47=b
        65=0,23=5,83=0,76=2,81=9,63=9,49=2,26=8,52=7,69=4,32=5,94=2
        02=2,20=2,12=0,11=2,02=2,10=1,12=0,01=1,01=1,20=2,01=1,12=0
        groups:
        CyclicGroup(13) CyclicGroup(3)
        CyclicGroup(11)
        CyclicGroup(3)
        orders:
        13 3
        11
        3
        vocabulary:
        "0123456789abcdef"
        "0123456789a"
        "012"
        fixed_vocabulary:
        "0123456789abcdef"
        "0123456789a"
        "012"
        prediction_mask:
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001''')
    a = MixDihedralGroupAddition(max_order=10, holdout_zero=True)
    assert eqstring(
        a.stringify(a.sample_batch(batch_size=3, k_shots=12)), '''
        inputs:
        ,22=2,cb=b,57=f,ba=3,ee=c,3e=4,ce=e,1e=a,ee=c,09=6,60=9,bc=
        ,c5=d,d6=f,6f=6,71=7,bb=1,6a=4,ea=c,77=1,6a=4,81=8,4f=4,ec=
        ,9d=3,d2=3,5b=5,a9=3,3e=1,d3=2,e2=9,ee=2,12=d,5e=b,e1=3,32=
        targets:
        22=2,cb=b,57=f,ba=3,ee=c,3e=4,ce=e,1e=a,ee=c,09=6,60=9,bc=b
        c5=d,d6=f,6f=6,71=7,bb=1,6a=4,ea=c,77=1,6a=4,81=8,4f=4,ec=a
        9d=3,d2=3,5b=5,a9=3,3e=1,d3=2,e2=9,ee=2,12=d,5e=b,e1=3,32=a
        groups:
        DihedralGroup(2) DihedralGroup(4) DihedralGroup(2)
        DihedralGroup(2) DihedralGroup(4)
        DihedralGroup(5)
        orders:
        4 8 4
        4 8
        10
        vocabulary:
        "2960c13eb4ad875f"
        "1b78f6ec4a5d"
        "b25d3a41e9"
        fixed_vocabulary:
        ""
        ""
        ""
        prediction_mask:
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001''')
    a = MixCyclicGroupAddition(max_order=10, holdout_zero=True)
    assert eqstring(
        a.stringify(a.sample_batch(batch_size=3, k_shots=12, fixed_p=1.0, hold_out=1)), '''
        inputs:
        ,23=1,97=5,10=1,03=3,23=1,58=9,5a=4,aa=9,03=3,dc=b,56=7,cc=
        ,46=1,da=e,aa=b,81=0,45=0,63=0,a9=a,31=4,27=0,bd=f,28=1,01=
        ,65=3,02=2,60=6,36=1,44=0,07=7,75=4,31=4,05=5,45=1,23=5,41=
        targets:
        23=1,97=5,10=1,03=3,23=1,58=9,5a=4,aa=9,03=3,dc=b,56=7,cc=d
        46=1,da=e,aa=b,81=0,45=0,63=0,a9=a,31=4,27=0,bd=f,28=1,01=1
        65=3,02=2,60=6,36=1,44=0,07=7,75=4,31=4,05=5,45=1,23=5,41=5
        groups:
        CyclicGroup(4) CyclicGroup(7) CyclicGroup(3)
        CyclicGroup(9) CyclicGroup(7)
        CyclicGroup(8)
        orders:
        4 7 3
        9 7
        8
        vocabulary:
        "0123456789abcd"
        "0123456789abcdef"
        "01234567"
        fixed_vocabulary:
        "0123456789abcd"
        "0123456789abcdef"
        "01234567"
        prediction_mask:
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001
        00010000100001000010000100001000010000100001000010000100001''')

    # --- structural invariants (model-free): fixed_p, min_order, constant-group,
    # --- and max_length enforcement. These don't depend on golden RNG strings, so they
    # --- use their own task instances and can grow without re-deriving expected output.

    # Constant-group setup: min_order == max_order and mix=0 -> exactly one order-10
    # cyclic group every run, so total_order is a constant 10 (each 0.1 step of fixed_p
    # then pins exactly one more element).
    c10 = MixCyclicGroupAddition(num_symbols=16, max_order=10, min_order=10, mix=0.0, seed=1)
    for _ in range(20):
        _, groups, orders, _ = c10.sample_run(k_shots=8)
        assert len(groups) == 1 and orders == [10], f"expected one C10, got {orders}"

    # _fixed_slots: exact count round(fixed_p*order), deterministic, nested as fixed_p grows.
    prev = set()
    for p in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        s = c10._fixed_slots(10, p)
        assert len(s) == round(p * 10), f"fixed_p={p}: |slots|={len(s)} != {round(p * 10)}"
        assert s == c10._fixed_slots(10, p), "fixed_slots not deterministic"
        assert prev <= s, f"fixed_p={p}: slots not nested over the smaller fixed_p"
        prev = s
    assert c10._fixed_slots(10, 0.0) == set(), "fixed_p=0 must pin nothing"
    assert c10._fixed_slots(10, 1.0) == set(range(10)), "fixed_p=1 must pin every slot"

    # fixed_vocabulary length in a batch == number of pinned slots.
    b = c10.sample_batch(batch_size=4, k_shots=8, fixed_p=0.3)
    assert all(len(fv) == round(0.3 * 10) for fv in b["fixed_vocabulary"]), b["fixed_vocabulary"]

    # min_order lower-bounds every sampled group's order.
    lo = MixCyclicGroupAddition(num_symbols=16, max_order=12, min_order=7, mix=0.5, seed=2)
    for _ in range(20):
        for G in lo.sample_groups():
            assert G.order() >= 7, f"min_order=7 violated: sampled order {G.order()}"

    # max_length is enforced: 5*k_shots-1 input tokens must fit, else a clear ValueError.
    try:
        c10.sample_batch(batch_size=1, k_shots=100, max_length=50)
        raise AssertionError("expected ValueError: k_shots too large for max_length")
    except ValueError:
        pass

    print("all _unit_test assertions passed")

if __name__ == '__main__':
    _unit_test()
