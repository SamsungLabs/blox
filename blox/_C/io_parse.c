#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <endian.h>
#include <stdio.h>
#include <math.h>

#define HASH 0
#define STR 1
#define FLOAT 2
#define ARCH 3
#define INT 4
#define QUANTIZED 5
#define LIST_BIT 0x08

#define CHECK_ENOUGH_BYTES(n) \
    do { \
        if ((n) > buffer_size) { \
            PyErr_SetString(PyExc_IndexError, "Buffer too short"); \
            goto error; \
        } \
    } while (0)

#define ADVANCE_BYTES(n) CHECK_ENOUGH_BYTES(n); ptr = buffer; buffer += (n); buffer_size -= (n)


static const char dec_to_hex[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
};


static PyObject* _fetch_quantized_values(PyObject* list, const char* buff, unsigned int buff_len, unsigned int list_len,
                                        unsigned char bits, uint16_t offset, uint16_t total_examples)
{
    uint64_t local = 0;
    uint64_t local2;
    unsigned long tmp;
    double d;
    PyObject* tmp_obj;
    unsigned char curr_len = 0, curr_offset = 0, consumed_bits = 0, consumed_bytes = 0;

    // printf("Decoding quantized list - bytes: %u, list lenght: %u, bits: %hhu, offset: %hu, total_examples: %hu\n", buff_len, list_len, bits, offset, total_examples);

    for (unsigned int i=0; i<list_len; ++i) {
        if (curr_len < bits) {
            if (buff_len <= 0) {
                PyErr_SetString(PyExc_AssertionError, "Not enough packed data?");
                return NULL;
            }
            if (buff_len >= sizeof(local2)) {
                // printf("Quick fetching new data... ");
                local2 = le64toh(*((const uint64_t*)buff));
                consumed_bits = sizeof(local2) * 8;
            } else {
                // printf("Fetching last bytes... ");
                local2 = 0;
                for (unsigned int j=0; j<buff_len; ++j) {
                    local2 |= ((uint64_t)(*((const unsigned char*)buff+j))) << (j*8);
                }
                consumed_bits = buff_len * 8;
            }
            local2 >>= curr_offset;
            local2 <<= curr_len;
            local |= local2;

            // printf("Consumed %hhu raw bits, with offset %hhu, and merged with current data of length %hhu ", consumed_bits, curr_offset, curr_len);

            if (curr_len > curr_offset)
                consumed_bits -= (curr_len - curr_offset);

            // printf("Effectively consumed bits: %hhu ", consumed_bits);

            curr_len += consumed_bits - curr_offset;
            curr_offset = consumed_bits % 8;

            consumed_bytes = consumed_bits / 8;
            buff += consumed_bytes;
            buff_len -= consumed_bytes;

            // printf("Current length: %hhu, offset: %hhu, consumed bytes: %hhu, remaining bytes: %u\n", curr_len, curr_offset, consumed_bytes, buff_len);
        }

        tmp = (local & ((1 << bits) - 1));
        local >>= bits;
        curr_len -= bits;
        d = (((double)tmp + offset) / total_examples * 100);
        d = round(d * 1e3) / 1e3;
        tmp_obj = PyFloat_FromDouble(d);
        if (tmp_obj == NULL)
            return NULL;

        PyList_SET_ITEM(list, i, tmp_obj);
    }

    return list;
}



// python API: f(bytes, num_rows, list_of_encoded_dtypes)
static PyObject* io_parse(PyObject* self, PyObject* args)
{
    const char* buffer;
    Py_ssize_t buffer_size;
    const char* ptr;
    unsigned int rows;
    PyObject* dtypes;
    Py_ssize_t num_columns;

    PyObject* tmp_obj;
    PyObject* tmp_obj2;
    PyObject* tmp_obj3;
    PyObject* ret = NULL;

    //dtype encoding
    uint8_t type_bits, extra_bits, bits;
    uint16_t qoffset; 
    uint16_t examples;

    //for types with list bit set
    unsigned char is_list;
    unsigned int list_len;

    // helpers
    char model_hash_buffer[32];
    float f32;
    double f64;
    unsigned int u32;
    unsigned int total_bytes;

    // printf("AAAA!!!\n"); fflush(stdout);

    if (!PyArg_ParseTuple(args, "y#IO!", &buffer, &buffer_size, &rows, &PyList_Type, &dtypes))
        return NULL;

    // printf("BBBBBB %zi %u\n", buffer_size, rows); fflush(stdout);

    num_columns = PyList_Size(dtypes);

    // printf("Columns!!! %zi\n", num_columns); fflush(stdout);
    uint32_t* columns = (uint32_t*)malloc(sizeof(uint32_t) * num_columns);
    if (columns == NULL)
        return PyErr_NoMemory();

    // printf("Allocated!!!\n"); fflush(stdout);

    for (Py_ssize_t i=0; i<num_columns; ++i) {
        tmp_obj = PyList_GET_ITEM(dtypes, i);
        if (tmp_obj == NULL)
            goto error;
        columns[i] = (uint32_t)PyLong_AsUnsignedLong(tmp_obj);
        if (columns[i] == (uint32_t)-1)
            goto error;

        // printf("Column %i raw encoded dtype: %u\n", i, columns[i]); fflush(stdout);
    }

    ret = PyList_New((Py_ssize_t)rows);
    if (ret == NULL)
        goto error;

    for (unsigned int row=0; row<rows; ++row) {
        tmp_obj = PyList_New(num_columns);
        if (tmp_obj == NULL)
            goto error;
        PyList_SET_ITEM(ret, row, tmp_obj);

        for (unsigned int col=0; col<num_columns; ++col) {
            type_bits = columns[col] & 0x07;
            is_list = ((columns[col] & LIST_BIT) != 0);
            extra_bits = (columns[col] >> 4) & 0x0F;
            bits = (columns[col] >> 8) & 0xFF;
            qoffset = (columns[col] >> 16) & 0xFFFF;

            if (!row) {
                // printf("Column %i type: %hhi (list: %hhi) extra: %hhi bits: %hhi offset: %hi\n", col, type_bits, is_list, extra_bits, bits, qoffset);
            }

            if (is_list) {
                ADVANCE_BYTES(1);
                list_len = *((const unsigned char*)ptr);
                // printf("List len: %i\n", list_len);
                tmp_obj3 = PyList_New(list_len);
                if (tmp_obj3 == NULL)
                    goto error;
                PyList_SET_ITEM(tmp_obj, col, tmp_obj3);
            } else {
                list_len = 1;
                tmp_obj3 = NULL;
            }

            for (unsigned int within_list=0; within_list<list_len; ++within_list) {
                switch (type_bits) {
                    case HASH:
                        ADVANCE_BYTES(16);
                        for (int i=0; i<16; ++i) {
                            model_hash_buffer[31-(i*2)] = dec_to_hex[ptr[i] & 0x0F];
                            model_hash_buffer[31-(i*2+1)] = dec_to_hex[(ptr[i] >> 4) & 0x0F];
                        }
                        // printf("Model hash: %.32s\n", model_hash_buffer);
                        tmp_obj2 = PyUnicode_FromKindAndData(PyUnicode_1BYTE_KIND, model_hash_buffer, 32);
                        break;
                    case STR:
                        ADVANCE_BYTES(1);
                        u32 = (uint32_t)(*(unsigned char*)ptr);
                        ADVANCE_BYTES(u32);
                        tmp_obj2 = PyUnicode_FromStringAndSize(ptr, u32);
                        break;
                    case FLOAT:
                        ADVANCE_BYTES(3);
                        u32 = *((unsigned char*)ptr);
                        u32 |= (((uint32_t)(*((unsigned char*)(ptr+1)))) << 8);
                        u32 |= (((uint32_t)(*((unsigned char*)(ptr+2)))) << 16);
                        f64 = (double)u32 / 1e6;
                        tmp_obj2 = PyFloat_FromDouble(f64);
                        break;
                    case ARCH:
                        ADVANCE_BYTES(3);
                        tmp_obj2 = Py_BuildValue("[[[b][bbbb]][[b][bbbb]][[b][bbbb]]]",
                            (ptr[0]>>3) / 3,
                            (ptr[0]>>3) % 3,
                            (ptr[0]>>2) & 0x01,
                            (ptr[0]>>1) & 0x01,
                            ptr[0] & 0x01,
                            (ptr[1]>>3) / 3,
                            (ptr[1]>>3) % 3,
                            (ptr[1]>>2) & 0x01,
                            (ptr[1]>>1) & 0x01,
                            ptr[1] & 0x01,
                            (ptr[2]>>3) / 3,
                            (ptr[2]>>3) % 3,
                            (ptr[2]>>2) & 0x01,
                            (ptr[2]>>1) & 0x01,
                            ptr[2] & 0x01);
                        break;
                    case INT:
                        ADVANCE_BYTES(4);
                        u32 = le32toh(*((unsigned int*)ptr));
                        tmp_obj2 = PyLong_FromUnsignedLong(u32);
                        break;
                    case QUANTIZED:
                        if (within_list > 0) {
                            PyErr_SetString(PyExc_AssertionError, "quantized list reached second iteration");
                            goto error;
                        }

                        examples = (extra_bits == 0 ? 5000 : (extra_bits == 1 ? 10000 : 45000));
                        total_bytes = (bits * list_len + 7) / 8;
                        ADVANCE_BYTES(total_bytes);

                        if (tmp_obj3 == NULL) {
                            u32 = 0;
                            if (total_bytes > 4) {
                                PyErr_SetString(PyExc_AssertionError, "more than 32 bits for a single-element quantized value!");
                                goto error;
                            }
                            for (unsigned int i=0; i<total_bytes; ++i) {
                                u32 |= (((uint32_t)(*((const uint8_t*)ptr+i))) << ((uint8_t)(8*i)));
                            }
                            f64 = (((double)u32 + qoffset) / examples * 100);
                            f64 = round(f64 * 1e3) / 1e3;
                            tmp_obj2 = PyFloat_FromDouble(f64);
                        } else {
                            tmp_obj3 = _fetch_quantized_values(tmp_obj3, ptr, total_bytes, list_len, bits, qoffset, examples);
                            within_list = list_len;
                            if (tmp_obj3 == NULL)
                                goto error;
                        }
                        break;

                    default:
                        PyErr_SetString(PyExc_ValueError, "Invalid column data type");
                        goto error;
                }

                if (within_list < list_len && tmp_obj2 == NULL)
                    goto error;

                if (tmp_obj3 != NULL) {
                    if (within_list < list_len) // does not hold if we've just processed a list of packed ints in _fetch_quantized_values
                        PyList_SET_ITEM(tmp_obj3, within_list, tmp_obj2);
                }
                else
                    PyList_SET_ITEM(tmp_obj, col, tmp_obj2);
            }
        }
    }

    return ret;

error:
    free(columns);
    if (ret != NULL)
        Py_DECREF(ret);
    return NULL;
}


static PyMethodDef BloxCMethods[] = {
    {"parse",  io_parse, METH_VARARGS, "Parse data from a custom-encoded blox dataset."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef bloxCModule = {
    PyModuleDef_HEAD_INIT,
    "blox.C",
    NULL,
    -1,
    BloxCMethods
};

PyMODINIT_FUNC
PyInit_C(void)
{
    return PyModule_Create(&bloxCModule);
}
