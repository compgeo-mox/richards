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