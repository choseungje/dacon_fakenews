import numpy as np


def electra_padding(inputs, pad_token, pad_length=0, pad=True, pair=False):
    """
    1차원 배열에 대한 padding, segmentation 수행

    inputs 의 값 : ex) [[1, 4, 3, 4, 5, 3, 7], [2, 2, 3, 4, 1]]
    len(inputs) = 1 or 2
    point_token : 온점(.)
    pad_token : [PAD]
    pad_length : padding 할 길이
    pair : 문장의 segmentation 여부

    return
    pad_ids : 패딩된 토큰
    seq_length : 패딩 이전 문장 토큰의 길이
    segment_ids : segment token
    """

    # print(inputs)
    pad_ids = []
    segment_ids = []

    # 하나의 문장만 들어올 때
    if len(inputs) == 1:
        # pad_length > len(inputs[0])
        if pad_length > len(inputs[0]):
            seq_length = len(inputs[0])

            if pad is True:
                pad_seq = (pad_length - len(inputs[0])) * [pad_token]
                inputs[0].extend(pad_seq)

            pad_ids.extend(inputs[0])

        # pad_length < len(inputs[0])
        else:
            inputs = inputs[0]

            if pad is True:
                # 토큰 리스트 슬라이싱
                inputs = inputs[:pad_length]

            # 문장 토큰의 마지막엔 3 추가
            inputs[-1] = 3
            pad_ids.extend(inputs)
            seq_length = len(pad_ids)

        # segment ids
        segment_ids.extend([0] * len(pad_ids))

    # 두개의 문장이 들어올 때
    else:
        # pair == True
        if pair is True:
            # pad_length > len(inputs[0]) + len(inputs[1][1:])
            if pad_length > len(inputs[0]) + len(inputs[1][1:]):
                seq_length = len(inputs[0]) + len(inputs[1][1:])

                pad_ids.extend(inputs[0])
                pad_ids.extend(inputs[1][1:])

                segment_ids.extend([0] * len(inputs[0]))
                segment_ids.extend([1] * len(inputs[1][1:]))

                if pad is True:
                    pad_seq = [pad_token] * (pad_length - seq_length)
                    pad_ids.extend(pad_seq)
                    segment_ids.extend(pad_seq)

            # pad_length < len(inputs[0]) + len(inputs[1][1:])
            else:
                pad_ids.extend(inputs[0])
                pad_ids.extend(inputs[1][1:])

                segment_ids.extend([0] * len(inputs[0]))
                segment_ids.extend([1] * len(inputs[1][1:]))

                if pad is True:
                    pad_ids = pad_ids[:pad_length]
                    segment_ids = segment_ids[:pad_length]
                pad_ids[-1] = 3

                seq_length = len(pad_ids)

        # fair == False
        else:
            inputs[0].extend(inputs[1][1:])
            inputs = inputs[0]

            # pad_length > len(inputs[0])
            if pad_length > len(inputs):
                seq_length = len(inputs)

                pad_ids.extend(inputs)
                if pad is True:
                    pad_ids.extend([0] * (pad_length - len(pad_ids)))

                segment_ids.extend([0] * len(pad_ids))

            # pad_length < len(inputs[0])
            else:
                pad_ids.extend(inputs)
                if pad is True:
                    pad_ids = pad_ids[:pad_length]
                pad_ids[-1] = 3

                segment_ids.extend([0] * len(pad_ids))

                seq_length = len(pad_ids)

    return np.array(pad_ids), np.array(seq_length), np.array(segment_ids)


if __name__ == "__main__":
    electra_padding([[1, 4, 3, 4, 5, 2, 7]], 0, 6, pair=True)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 7]], 0, 10, pair=True)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 3]], 0, pad=False, pair=True)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 3]], 0, pad=False, pair=False)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 7], [2, 2, 5, 4, 1]], 0, 6, pair=True)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 7], [2, 2, 5, 4, 1]], 0, 6, pair=False)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 7], [2, 2, 5, 4, 3]], 0, pad=False, pair=True)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 7], [2, 2, 5, 4, 3]], 0, 15, pad=True, pair=True)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 7], [2, 2, 5, 4, 3]], 0, 9, pad=True, pair=True)
    print()
    electra_padding([[1, 4, 3, 4, 5, 2, 7], [2, 2, 5, 4, 3]], 0, 9, pad=True, pair=False)
    print()

