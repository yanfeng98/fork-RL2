import heapq
from typing import List

def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    class Set:

        def __init__(self) -> None:
            self.sum: int = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other: "Set"):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other: "Set") -> bool:
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k: int = k
            self.sets: list[Set] = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self) -> list[list[int]]:
            partitions: list[list[int]] = []
            for i in range(len(self.sets)):
                cur_partition: list[int] = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other: "State"):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other: "State") -> bool:
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str: str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list: list[tuple[int, int]] = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq: list[State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items: list[tuple[int, int]] = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0: State = heapq.heappop(states_pq)
        state1: State = heapq.heappop(states_pq)
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state: State = states_pq[0]
    partitions: list[list[int]] = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * \
                k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions

def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """ get order of seq lengths to make partitions balanced, this is
        used in balacing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions: List[List[int]]) -> list[list[int]]:
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions: list[list[int]] = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions: list[list[int]] = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)