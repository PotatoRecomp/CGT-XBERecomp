"""Copies one file off the root of an Xbox disc image (XDVDFS), for build.sh to hand the xbe to the recompiler.

usage: xiso_get.py <image.iso> <name> <out>

The image may be a whole disc dump (redump, game partition at one of the known offsets) or the game partition alone.
"""
import struct
import sys

SECTOR = 2048
MAGIC = b"MICROSOFT*XBOX*MEDIA"
PARTITIONS = (0, 0x18300000, 0x30600000, 0x1FB20000, 0x0FD90000)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    iso, name, out = sys.argv[1:]
    with open(iso, "rb") as f:
        for base in PARTITIONS:
            f.seek(base + 0x10000)
            header = f.read(0x20)
            if header[: len(MAGIC)] == MAGIC:
                break
        else:
            sys.exit(f"{iso}: no Xbox file system")
        root, size = struct.unpack_from("<II", header, len(MAGIC))
        f.seek(base + root * SECTOR)
        table = f.read(size)

        # the root directory is a binary tree of entries, offsets in dwords
        pending = [0]
        while pending:
            off = pending.pop() * 4
            if off + 14 > len(table):
                continue
            left, right, sector, length, _attr, name_len = struct.unpack_from("<HHIIBB", table, off)
            entry = table[off + 14 : off + 14 + name_len].decode("latin1")
            if entry.lower() == name.lower():
                f.seek(base + sector * SECTOR)
                data = f.read(length)
                if len(data) != length:
                    sys.exit(f"{iso}: {name} is cut short")
                with open(out, "wb") as o:
                    o.write(data)
                print(f"{name}: {length} bytes")
                return
            pending += [c for c in (left, right) if c not in (0, 0xFFFF)]
    sys.exit(f"{iso}: no {name} in the root")


if __name__ == "__main__":
    main()
