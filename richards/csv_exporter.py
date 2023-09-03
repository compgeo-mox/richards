import os



class Csv_Exporter:
    def __init__(self, export_directory, file_name, headers):
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        if os.path.exists(os.path.join(export_directory, file_name)):
            print('Csv_Exporter: A file with the same name is detected. I\'ll delete it')
            os.remove(os.path.join(export_directory, file_name))

        self.path = os.path.join(export_directory, file_name)
        
        with open(self.path, 'w') as f:
            for i in range(len(headers)-1):
                f.write(str(headers[i]) + ',')
            f.write(str(headers[-1]) + '\n')
    
    def add_entry(self, entries):
        with open(self.path, 'a') as f:
            for i in range(len(entries)-1):
                f.write(str(entries[i]))
                f.write(',')
            f.write(str(entries[-1]) + '\n')