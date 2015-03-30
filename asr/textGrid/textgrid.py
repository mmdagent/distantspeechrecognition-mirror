import string

class TextInterval:

    def __init__(self, fp):
        for idx in range(4):
            line = string.strip(fp.next())
            items = string.split(line)
            if idx == 0:
                line = line.replace('intervals [', '')
                line = line.replace(']:', '')
                self.index = int(line)
            elif idx == 1:
                self.xmin = float(items[2])
            elif idx == 2:
                self.xmax = float(items[2])
            elif idx == 3:
                self.text = items[2]

    def write(self, fp):
        for idx in range(4):
            if idx == 0:
                fp.write('        intervals [%d]:\n' %self.index)
            elif idx == 1:
                fp.write('            xmin = %0.2f\n' %self.xmin)
            elif idx == 2:
                fp.write('            xmax = %0.2f\n' %self.xmax)
            elif idx == 3:
                fp.write('            text = %s\n' %self.text)


class TextTier:

    def __init__(self, fp):
        
        # Read the header information
        for idx in range(6):
            line = string.strip(fp.next())
            items = string.split(line)
            if idx == 0:
                line = line.replace('item [', '')
                line = line.replace(']:', '')
                self.index = int(line)
            elif idx == 1:
                self.classs = items[2]
            elif idx == 2:
                self.name = items[2]
            elif idx == 3:
                self.xmin = float(items[2])
            elif idx == 4:
                self.xmax = float(items[2])
            elif idx == 5:
                self.size = int(items[3])

        # Now read the individual intervals
        self.intervals = []
        for idx in range(self.size):
            self.intervals.append(TextInterval(fp))

    def write(self, fp):

        # Write the header information
        for idx in range(6):
            if idx == 0:
                fp.write('    item [%d]:\n' %self.index)
            elif idx == 1:
                fp.write('        class = %s\n' %self.classs)
            elif idx == 2:
                fp.write('        name = %s\n' %self.name)
            elif idx == 3:
                fp.write('        xmin = %0.2f\n' %self.xmin)
            elif idx == 4:
                fp.write('        xmax = %0.2f\n' %self.xmax)
            elif idx == 5:
                fp.write('        intervals: size = %d\n' %self.size)

        # Now write the individual intervals
        for interval in self.intervals:
            interval.write(fp)


class TextGrid:

    def __init__(self, fileName = ''):
        if fileName == '':
            self.clear()
        else:
            self.read(fileName)

    def clear(self):
        self.type  = ''
        self.classs = ''
        self.xmin  = 0.0
        self.xmax  = 0.0
        self.tiers = []
        self.size  = 0

    def read(self, fileName):
        self.clear()

        fp = open(fileName, 'r')

        # Read the header information
        idx = 0
        for line in fp:
            items = string.split(string.strip(line))
            if idx == 0:
                self.type  = items[3]
            elif idx == 1:
                self.classs = items[3]
            elif idx == 3:
                self.xmin = float(items[2])
            elif idx == 4:
                self.xmax = float(items[2])
            elif idx == 6:
                self.size = int(items[2])
            elif idx == 7:
                break

            idx += 1

        # Now read the individual tiers
        for n in range(self.size):
            self.tiers.append(TextTier(fp))

        fp.close()
            
    def __iadd__(self, other):
        for tier in other.tiers:
            self.tiers.append(tier)
            self.size += 1
            tier.index = self.size

        if other.xmin < self.xmin:
            self.xmin = other.xmin

        if other.xmax > self.xmax:
            self.xmax = other.xmax

        return self

    def write(self, fileName):

        # Write the header information
        fp = open(fileName, 'w')
        for idx in range(8):
            if idx == 0:
                fp.write('File type = %s\n' %self.type)
            elif idx == 1:
                fp.write('Object class = %s\n' %self.classs)
            elif idx == 2:
                fp.write('\n')
            elif idx == 3:
                fp.write('xmin = %0.2f\n' %self.xmin)
            elif idx == 4:
                fp.write('xmax = %0.2f\n' %self.xmax)
            elif idx == 5:
                fp.write('tiers? <exists>\n')
            elif idx == 6:
                fp.write('size = %d\n' %self.size)
            elif idx == 7:
                fp.write('item []:\n')

        # Now write the individual tiers
        for tier in self.tiers:
            tier.write(fp)
        fp.close()
