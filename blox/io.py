import io
import csv
import math
import pickle
import struct
import zlib

import yaml
import numpy as np

from . import utils

try:
    from . import C
    has_C = True
except:
    has_C = False

class BitPacker():
    def __init__(self, buff=None, total_bits=None):
        if buff is None:
            buff = bytearray()
        self.buff = buff
        self.rpos = [0, 0] # byte idx, bit idx (inside the relevant byte)

        if total_bits is None:
            total_bits = len(self.buff) * 8

        self.wpos = total_bits % 8 # offset in the last byte
        self.total_bits = total_bits

    def __str__(self):
        bits = []
        rem = self.total_bits
        for b in self.buff:
            byte = ''
            for _ in range(8):
                if b & 0x01:
                    byte += '1'
                else:
                    byte += '0'
                rem -= 1
                if not rem and _ < 7:
                    byte += '('

                b >>= 1

            if rem < 0:
                byte += ')'

            bits.append(byte)

        bits = ' '.join(bits)
        return f'BitPacker(bytes={len(self.buff)}, bits={self.total_bits}, rpos={self.rpos}, wpos={self.wpos}, content={bits})'

    def read(self, bits):
        if bits <= 0:
            raise ValueError('Positive integer expected')

        if self.rpos[0] * 8 + self.rpos[1] + bits > self.total_bits:
            raise ValueError('Not enough data in the buffer')

        first_byte = self.rpos[0]
        last_byte = self.rpos[0] + (self.rpos[1] + bits - 1) // 8

        ret = self.buff[first_byte:last_byte+1]
        ret = int.from_bytes(ret, 'little', signed=False)
        ret >>= self.rpos[1]
        ret &= ((1 << bits) - 1)
        ret = ret.to_bytes(((bits + 7) // 8), 'little', signed=False)

        # reuse functionality to increment rpos
        self.skip_read(bits)

        return ret

    def write(self, data, bits):
        if isinstance(data, int):
            if data < 0 or data > int(2**bits - 1):
                raise ValueError('Invalid integer')

        if self.wpos:
            if not isinstance(data, int):
                data = int.from_bytes(data, 'little', signed=False)
            data <<= self.wpos
            data = data.to_bytes((self.wpos + bits + 7) // 8, 'little', signed=False)
        else:
            if isinstance(data, int):
                data = data.to_bytes(((bits + 7) // 8), 'little', signed=False)

        if self.wpos:
            self.buff[-1] |= data[0]
            self.buff.extend(data[1:])
        else:
            self.buff.extend(data)

        self.total_bits += bits
        self.wpos = (self.wpos + bits) % 8

    def pad_write(self):
        self.total_bits += self.wpos
        self.wpos = 0

    def pad_read(self):
        if self.rpos[1] > 0:
            if (self.rpos[0]+1) * 8 > self.total_bits:
                raise ValueError('Not enough data')

            self.rpos[0] += 1
            self.rpos[1] = 0

    def skip_read(self, bits):
        if self.rpos[0] * 8 + self.rpos[1] + bits > self.total_bits:
            raise ValueError('Run out of content')

        byte_offset = (self.rpos[1] + bits) // 8
        bit_offset = (self.rpos[1] + bits) % 8
        self.rpos[0] += byte_offset
        self.rpos[1] = bit_offset

    def skip_write(self, bits):
        new_bytes = (self.wpos + bits) // 8
        self.buff.extend(b'\0' * new_bytes)
        self.wpos = (self.wpos + bits) % 8
        self.total_bits += bits


def _decode_slow(hdr, data, bits, rows):
    packer = BitPacker(data, bits)

    def _decode_field(t, cname):
        if t == 'hash':
            i = int.from_bytes(packer.read(16*8), 'little', signed=False)
            return hex(i)[2:].zfill(32)
        elif t == 'str':
            l = int.from_bytes(packer.read(8), 'little', signed=False)
            return packer.read(l*8).decode('ascii')
        elif t == 'float':
            i = struct.unpack('<I', packer.read(24))[0]
            return float(i) / 1e6
        elif t == 'arch':
            return [_decode_field('cell', cname) for _ in range(3)]
        elif t == 'cell':
            i = int.from_bytes(packer.read(8), 'little', signed=False)
            a = [[None], [None, None, None, None]]
            a[1][3] = bool(i & 0x01)
            a[1][2] = bool((i >> 1) & 0x01)
            a[1][1] = bool((i >> 2) & 0x01)
            a[1][0] = int((i >> 3) % 3)
            a[0][0] = int((i >> 3) // 3)
            return a
        elif t == 'int':
            return struct.unpack('<I', packer.read(32))[0]
        elif t.startswith('b_'):
            parts = t.split('_')
            minv = int(parts[1])
            bits = int(parts[3])
            num_examples = 45000 if 'train' in cname else (5000 if 'val' in cname else 10000)
            v = int.from_bytes(packer.read(bits), 'little', signed=False)
            v += minv
            acc = round(v / num_examples * 100, 3)
            return acc
        elif t.startswith('list['):
            assert t[-1] == ']'
            subt = t[5:-1]
            l = int.from_bytes(packer.read(8), 'little', signed=False)
            return [_decode_field(subt, cname) for _ in range(l)]
        else:
            raise ValueError('Unknown data type! ' + t)

    def _decode_row():
        ret = []
        for cname in hdr['columns']:
            dtype = hdr['column_types'][cname]
            ret.append(_decode_field(dtype, cname))
        return ret

    data = []
    for _ in range(rows):
        data.append(_decode_row())

    return data


def _decode_fast(hdr, data, rows):
    if not has_C:
        raise RuntimeError('Fast decoding not supported - could not load C backend, reinstalling blox package might fix this issue')

    encoded_dtypes = []
    for cname in hdr['columns']:
        dtype = hdr['column_types'][cname]
        list_bit = 0
        if dtype.startswith('list['):
            list_bit = 0x08
            dtype = dtype[5:-1]

        if dtype == 'hash':
            encoded_dtypes.append(0 | list_bit)
        elif dtype == 'str':
            encoded_dtypes.append(1 | list_bit)
        elif dtype == 'float':
            encoded_dtypes.append(2 | list_bit)
        elif dtype == 'arch':
            encoded_dtypes.append(3 | list_bit)
        elif dtype == 'int':
            encoded_dtypes.append(4 | list_bit)
        elif dtype.startswith('b_'):
            parts = dtype.split('_')
            qoffset = int(parts[1])
            bits = int(parts[3])
            examples = 2 if 'train' in cname else (0 if 'val' in cname else 1) # 0 -- val, 1 -- test, 2 -- train
            encoded = (5 | list_bit | (examples << 4) | (bits << 8) | (qoffset << 16))
            assert encoded.bit_length() < 32, 'cannot encode dtype with 32 bits!'
            encoded_dtypes.append(encoded)
        else:
            raise ValueError(f'Unknown type! {dtype!r} for column: {cname!r}')

    return C.parse(data, rows, encoded_dtypes)


def decode(hdr, data, fast=True):
    data = zlib.decompress(data)
    data1, data2 = data[:12], data[12:]
    bits, rows = struct.unpack('<QI', data1)
    bits -= 4*8

    if not fast:
        return _decode_slow(hdr, data2, bits, rows)
    else:
        if has_C:
            return _decode_fast(hdr, data2, rows)
        else:
            import warnings
            warnings.warn('Fast decoding is unsupported (could not load C backend), falling back to slow decoding', RuntimeWarning)
            return _decode_slow(hdr, data2, bits, rows)

def encode(hdr, data):
    dtypes = {}
    for idx, cname in enumerate(hdr['columns']):
        if cname == 'model_hash':
            dtypes[cname] = 'hash'
            continue
        if cname == 'arch_vec':
            dtypes[cname] = 'arch'
            continue
        if cname not in ['test_top1', 'val_top1', 'train_top1', 'train_top5', 'test_top5', 'val_top5']:
            if isinstance(data[0][idx], list):
                dtypes[cname] = f'list[{type(data[0][idx][0]).__name__}]'
            else:
                dtypes[cname] = type(data[0][idx]).__name__
            continue

        column_data = np.asarray([row[idx] for row in data])
        num_examples = 45000 if 'train' in cname else (5000 if 'val' in cname else 10000)
        correct = (column_data * num_examples).round().astype(np.int32) // 100
        assert np.allclose((correct / num_examples * 100), column_data)

        minv, maxv = correct.min(), correct.max()
        rv = maxv - minv
        bits_range = int(math.ceil(math.log2(rv)))
        if 'test' in cname and bits_range % 8 != 0:
            bits_range += (8 - (bits_range % 8))
        dtype = f'b_{minv}_{maxv}_{bits_range}'
        if isinstance(data[0][idx], list):
            dtype = f'list[{dtype}]'
        dtypes[cname] = dtype

    hdr['column_types'] = dtypes

    packer = BitPacker()

    def _encode_field(f, t, cname):
        if t == 'hash':
            i = int(f, 16)
            packer.write(i, 16*8)
        elif t == 'str':
            packer.write(len(f), 8)
            packer.write(f.encode('ascii'), len(f)*8)
        elif t == 'float':
            v = int(round(f, 6) * 1e6)
            f_ref = round(f, 6)
            if float(v) / 1e6 != f_ref:
                v += 1
                if float(v) / 1e6 != f_ref:
                    v -= 2
                    if float(v) / 1e6 != f_ref:
                        assert False, f'{f} {f_ref} {v+1}'
            assert v.bit_length() <= 24
            packer.write(v, 24)
        elif t == 'arch':
            assert len(f) == 3
            for a in f:
                _encode_field(a, 'cell', cname)
        elif t == 'cell':
            f = utils.flatten(f)
            i = ((f[0]*3 + f[1]) << 3) | (f[2] << 2) | (f[3] << 1) | f[4]
            packer.write(i, 8)
        elif t == 'int':
            b = struct.pack('<I', f)
            packer.write(b, len(b)*8)
        elif t.startswith('b_'):
            parts = t.split('_')
            minv = int(parts[1])
            bits = int(parts[3])
            num_examples = 45000 if 'train' in cname else (5000 if 'val' in cname else 10000)
            correct = int(round(f * num_examples)) // 100
            v = correct - minv
            packer.write(v, bits)
        elif t.startswith('list['):
            assert t[-1] == ']'
            assert len(f) < 256
            packer.write(len(f), 8)
            subt = t[5:-1]
            for ff in f:
                _encode_field(ff, subt, cname)
        else:
            raise ValueError('Unknown data type! ' + t)

    def _encode_row(row):
        for cname, data in zip(hdr['columns'], row):
            dtype = dtypes[cname]
            _encode_field(data, dtype, cname)
            packer.pad_write()

    packer.write(struct.pack('<I', len(data)), 32)
    for row in data:
        _encode_row(row)

    bits = packer.total_bits
    ret = struct.pack('<Q', bits) + packer.buff
    return zlib.compress(ret)


def parse_csv(hdr, data):
    cols = hdr['columns']
    for row in range(len(data)):
        for col in range(len(cols)):
            cname = cols[col]
            val = data[row][col]
            if 'top' in cname or 'loss' in cname:
                if 'test' not in cname:
                    data[row][col] = [float(v) for v in val[1:-1].split(',')]
                else:
                    data[row][col] = float(val)
            elif cname in ['train_time_s', 'params']:
                data[row][col] = int(val)
            elif cname in ['flops']:
                data[row][col] = float(val)
            elif cname == 'arch_vec':
                val = [int(v) for v in val.replace('[', '').replace(']', '').split(',')]
                data[row][col] = [
                    [[val[0]], [val[1], val[2], val[3], val[4]]],
                    [[val[5]], [val[6], val[7], val[8], val[9]]],
                    [[val[10]], [val[11], val[12], val[13], val[14]]]
                ]

    return data


def read_dataset_file(filename, fast=True):
    with open(filename, 'rb') as f:
        hdr_buff = f.read(1024)
        if not hdr_buff.startswith(b'---\n'):
            raise ValueError(f'Invalid dataset file: {filename!r}')

        hdr_end = hdr_buff.find(b'---\n', 3)
        if hdr_end == -1:
            raise ValueError(f'Could not find the header end in the first 1024 bytes of the dataset file: {filename!r}! Corrupted file?')

        f_hdr = hdr_buff[:hdr_end]
        f_rest = hdr_buff[hdr_end+4:] + f.read()
        del hdr_buff

    f_hdr = io.TextIOWrapper(io.BytesIO(f_hdr))
    hdr = yaml.load(f_hdr, Loader=yaml.SafeLoader)

    if hdr['format'] == 'csv':
        f_rest = io.TextIOWrapper(io.BytesIO(f_rest), encoding='utf-8', newline='')

    if hdr['format'] == 'csv':
        data = list(csv.reader(f_rest))
        data = parse_csv(hdr, data)
    elif hdr['format'] == 'pickle':
        data = pickle.loads(f_rest)
    elif hdr['format'] == 'custom':
        data = decode(hdr, f_rest, fast=fast)
    else:
        raise ValueError(hdr['format'])

    return hdr, data
