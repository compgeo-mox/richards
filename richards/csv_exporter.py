class Csv_Exporter:
    def __init__(self, headers):
        self.data = []
        self.headers = headers

    def add_entry(self, entry):
        self.data.append(entry)

    def export_file(self, export_path):
        with open(export_path, 'w') as f:
            f.write(self.headers)
            f.write('\n')

            for line in self.data:
                f.write(line + '\n')