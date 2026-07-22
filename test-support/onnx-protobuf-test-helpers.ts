/**
 * Minimal dependency-free ONNX protobuf readers for Python snippets executed
 * inside Pyodide integration tests. Keep operation-specific assertions in the
 * owning test; this shared prelude owns only wire traversal and common fields.
 */
export const ONNX_PROTOBUF_TEST_HELPERS = String.raw`
def fields(data):
    index = 0
    while index < len(data):
        tag = 0
        shift = 0
        while True:
            byte = data[index]
            index += 1
            tag |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        number = tag >> 3
        wire = tag & 0x7
        if wire == 0:
            value = 0
            shift = 0
            while True:
                byte = data[index]
                index += 1
                value |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    break
                shift += 7
            yield number, wire, value
        elif wire == 2:
            length = 0
            shift = 0
            while True:
                byte = data[index]
                index += 1
                length |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    break
                shift += 7
            payload = data[index:index + length]
            index += length
            yield number, wire, payload
        else:
            raise RuntimeError("unexpected protobuf wire type " + str(wire))

def attribute_map(node_fields):
    result = {}
    for field, kind, payload in node_fields:
        if field != 5 or kind != 2:
            continue
        attribute_fields = list(fields(payload))
        name = next(
            value.decode("utf-8") for child, child_kind, value in attribute_fields
            if child == 1 and child_kind == 2
        )
        result[name] = next(
            value for child, child_kind, value in attribute_fields
            if child == 3 and child_kind == 0
        )
    return result

def value_info_dtype(value_info):
    type_proto = next(
        payload for number, wire, payload in fields(value_info)
        if number == 2 and wire == 2
    )
    tensor_type = next(
        payload for number, wire, payload in fields(type_proto)
        if number == 1 and wire == 2
    )
    return next(
        value for number, wire, value in fields(tensor_type)
        if number == 1 and wire == 0
    )
`;
