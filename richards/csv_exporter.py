import os



class Csv_Exporter:
    """
    Simple class that can be used to construct a CSV file. 
    The headers must be provided when it is constructed.
    Entries may be added by calling the add_entry method.
    """
    def __init__(self, export_directory, file_name, headers, overwrite_existing=True):
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        if os.path.exists(os.path.join(export_directory, file_name)):
            print('Csv_Exporter: A file with name ' + str(file_name) + ' is detected. I\'ll delete it')
            os.remove(os.path.join(export_directory, file_name))

        self.path = os.path.join(export_directory, file_name)
        
        if overwrite_existing or not os.path.exists(self.path):
            self.file = open(self.path, 'w')
            for i in range(len(headers)-1):
                self.file.write(str(headers[i]) + ',')
            self.file.write(str(headers[-1]) + '\n')
        
    def add_entry(self, entries):
        self.file.flush()

        for i in range(len(entries)-1):
            self.file.write(str(entries[i]))
            self.file.write(',')
        self.file.write(str(entries[-1]) + '\n')

    def __del__(self):
        self.file.close()