import os

class Csv_Exporter:
    def __init__(self, *headers):
        self.data = []
        self.headers = ''
        
        for i in range(len(headers)-1):
            self.headers = self.headers + str(headers[i]) + ','
        self.headers = self.headers + str(headers[-1])
    
    def add_entry(self, *entries):
        final_entry = ''
        
        for i in range(len(entries)-1):
            final_entry = final_entry + str(entries[i]) + ','
        self.data.append(final_entry + str(entries[-1]))

    def export_file(self, export_directory, file_name):
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        if os.path.exists(os.path.join(export_directory, file_name)):
            print('Csv_Exporter: A file with the same name is detected. I\'ll delete it')
            os.remove(os.path.join(export_directory, file_name))


        with open(os.path.join(export_directory, file_name), 'w') as f:
            f.write(self.headers)
            f.write('\n')

            for line in self.data:
                f.write(line + '\n')