import copy
import csv 
#from tabulate import tabulate

#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        '''
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                print(self.data[i][j], end=" ")
            print()
        '''
        #print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.data[0]) # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier)
        col = []
        for row in self.data: 
            if include_missing_values:
                col.append(row[col_index])
            else: 
                if row[col_index] != "NA":
                    col.append(row[col_index])
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        
        # loops through each value in table to convert to numeric
        # if not able to convert, string is printed to terminal
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    numeric_value = float(self.data[i][j])
                    # success!!
                    self.data[i][j] = round(numeric_value,3)
                except ValueError:
                    # failure
                    continue

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.
        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        new_data = []

        #loops through rows in current data and adds rows not in rows_to_drop in new table
        for i in range(len(self.data)):
            if self.data[i] not in rows_to_drop and self.data[i] not in new_data:
                new_data.append(self.data[i])
        self.data = new_data

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename) as filename:
            csv_reader = csv.reader(filename, delimiter=',')
            self.data = list(csv_reader)
            self.column_names = self.data.pop(0)
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
                fileWriter = csv.writer(csvfile, delimiter=',')
                fileWriter.writerow(self.column_names)
                for row in self.data:
                     fileWriter.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        list_of_indexes = []
        test_row = []

        # adds corresponding indexes in curr table and key_collumn names to list
        for i in range(len(key_column_names)):
            for j in range(len(self.column_names)):
                if key_column_names[i] == self.column_names[j]:
                    list_of_indexes.append(j)
        
        # loops through current table
        for i in range(len(self.data)):
            test_row = []

            #loops through list of indexes and adds corresponding values to row to check duplicates
            for j in range(len(list_of_indexes)):
                test_row.append(self.data[i][list_of_indexes[j]])

            # checks each row in table for a duplicate of current test row
            for k in range(i+1, len(self.data)):

                #only one key column
                if len(key_column_names) == 1:
                    if test_row[0] in self.data[k]:
                        duplicates.append(self.data[k])

                #multiple key columns
                else:

                    #checks for duplicates and adds the duplicate to new list
                    if test_row[0] == self.data[k][list_of_indexes[0]] and test_row[1] == self.data[k][list_of_indexes[1]]:
                        duplicates.append(self.data[k])
        
        # checks if last row in duplicates is equil to row before
        if (len(duplicates) != 0 and len(duplicates) != 1):
            if duplicates[len(duplicates)-1] == duplicates[len(duplicates)-2]:
                del duplicates[len(duplicates)-2]


     
        return duplicates # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
       
        new_data = []
        
        #loops through table and adds all rows without NA values to new table
        for i in range(len(self.data)):
            if "NA" not in self.data[i]:
                new_data.append(self.data[i])

        #sets data to new data with no NA values
        self.data = new_data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        summ = 0
        length = 0
        col = self.get_column(col_name)
        col_index = -1

        #finds col_name index
        for i in range(len(self.column_names)):
            if self.column_names[i] == col_name:
                col_index = i

        # loops through each item in collum and computes average of continuous data
        for item in col:
            if (type(item) != str):
                summ += item
                length +=1
        if (length != 0):
            average = summ / length

        # finds NA values in collumn name and fills it with average value
        if (col_index != -1):
            for i in range(len(self.data)):
                if self.data[i][col_index] == "NA":
                    self.data[i][col_index] = round(average, 3)

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """

        #checks for empty table
        if not self.data:
            return MyPyTable()

        new_row = []
        new_table = []
        for cols in col_names:
            new_row = []
            curr_col = self.get_column(cols, False)

            #sorts row for finding median
            curr_col.sort()
            new_row.append(cols)

            #adds min, max, mid range, and mean
            new_row.append(min(curr_col))
            new_row.append(max(curr_col))
            new_row.append((min(curr_col) + max(curr_col)) / 2 )
            new_row.append(round(sum(curr_col) / len(curr_col), 3))
        
            # finds median for even number of instances and odds
            median = curr_col[len(curr_col)//2]
            median_2 = curr_col[len(curr_col) // 2 - 1]

            #odd
            if (len(curr_col) %2 != 0):
                    new_row.append(median)
            #even
            else:
                    new_row.append(((median + median_2) / 2))
            new_table.append(new_row)
        
        #creates new table and returns it
        new_table = MyPyTable(data=new_table)
        return new_table # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_list_of_indexes = []
        other_list_of_indexes = []
        merge_table_index = -1
        curr_row = []
        merge_table = []
        new_headers = []

        # appending new column names to new table
        for i in range(len(self.column_names)):
            new_headers.append(self.column_names[i])
        for i in range(len(other_table.column_names)):
            if (other_table.column_names[i] not in new_headers):
                new_headers.append(other_table.column_names[i])

        #creating list of valid indexes for current table
        for i in range(len(key_column_names)):
            for j in range(len(self.column_names)):
                if key_column_names[i] == self.column_names[j]:
                    self_list_of_indexes.append(j)
        
        #creating list of valid indexes for other table
        for i in range(len(key_column_names)):
            for j in range(len(other_table.column_names)):
                if key_column_names[i] == other_table.column_names[j]:
                    other_list_of_indexes.append(j)

        #looping through each row in current table
        for i in range(len(self.data)):
            curr_row = self.data[i]

            # loops through each row in other table
            for j in range(len(other_table.data)):

                # only one key column name
                if (len(key_column_names) == 1):

                    #checks if curr row in current table is a match with other_table and appends row
                    if (curr_row[self_list_of_indexes[0]] == other_table.data[j][other_list_of_indexes[0]]):
                        merge_table.append(curr_row)
                        merge_table_index+=1

                        #appends variables in curr row in other table to curr row in merge_table
                        for k in range(len(other_table.data[j])):
                            if other_table.data[j][k] not in merge_table[merge_table_index]:
                                merge_table[merge_table_index].append(other_table.data[j][k])
                                
                # more than one key column name
                else: 

                    #same method as above although with multiple keys to check for matches
                    if (curr_row[self_list_of_indexes[0]] == other_table.data[j][other_list_of_indexes[0]]) and (curr_row
                    [self_list_of_indexes[1]] == other_table.data[j][other_list_of_indexes[1]]):
                        merge_table.append(curr_row)
                        merge_table_index+=1

                        for k in range(len(other_table.data[j])):
                            if other_table.data[j][k] not in merge_table[merge_table_index]:
                                merge_table[merge_table_index].append(other_table.data[j][k])

        #creates new table and returns it
        newTable = MyPyTable(data=merge_table, column_names=new_headers)
        return newTable
            

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        self_list_of_indexes = []
        other_list_of_indexes = []
        merge_table_index = -1
        curr_row = []
        merge_table = []
        new_headers = []
        na_list = []
        no_matches = []
        list_of_valid_indexes = []
        
        #appends new column names to new headers
        for i in range(len(self.column_names)):
            new_headers.append(self.column_names[i])
        for i in range(len(other_table.column_names)):
            if (other_table.column_names[i] not in new_headers):
                new_headers.append(other_table.column_names[i])

        # creates list of valid indexes for current table
        for i in range(len(key_column_names)):
            for j in range(len(self.column_names)):
                if key_column_names[i] == self.column_names[j]:
                    self_list_of_indexes.append(j)
        
        # creates list of valid indexes for other table
        for i in range(len(key_column_names)):
            for j in range(len(other_table.column_names)):
                if key_column_names[i] == other_table.column_names[j]:
                    other_list_of_indexes.append(j)

        # loops through each row in current table
        for i in range(len(self.data)):
            single_match_found = False
            curr_row = self.data[i]

            # same method as inner join except adds rows in curr table without a match to a list
            for j in range(len(other_table.data)):

                # only one key column
                if (len(key_column_names) == 1):
                    if (curr_row[self_list_of_indexes[0]] == other_table.data[j][other_list_of_indexes[0]]):
                        list_of_valid_indexes.append(j)
                        merge_table.append(curr_row)
                        merge_table_index+=1
                        single_match_found = True
                        for k in range(len(other_table.data[j])):
                            if other_table.data[j][k] not in merge_table[merge_table_index]:
                                merge_table[merge_table_index].append(other_table.data[j][k])

                #multiple key columns
                else:
                    if (curr_row[self_list_of_indexes[0]] == other_table.data[j][other_list_of_indexes[0]]) and (curr_row[self_list_of_indexes[1]] == other_table.data[j][other_list_of_indexes[1]]):
                        list_of_valid_indexes.append(j)
                        merge_table.append(curr_row)
                        merge_table_index+=1
                        single_match_found = True
                        for k in range(len(other_table.data[j])):
                            if other_table.data[j][k] not in merge_table[merge_table_index]:
                                merge_table[merge_table_index].append(other_table.data[j][k])

            if (single_match_found == False):
                no_matches.append(curr_row)

        # loops trhough each instance in no_matches
        for i in range(len(no_matches)):

            # creates a list of NA values and appends it merge_table
            na_list = []
            for j in range(len(new_headers)):
                na_list.append("NA")
            merge_table.append(na_list)
            merge_table_index+=1

            # loops through current instance with no match and adds values to current NA list
            for j in range(len(no_matches[i])):
                for l in range(len(new_headers)):
                    if (new_headers[l] == self.column_names[j]):
                            merge_table[merge_table_index][l] = no_matches[i][j]

        # loops through each row in other table
        for i in range(len(other_table.data)):

            #checks through list of valid indexes to find rows without matches in other table
            if i not in list_of_valid_indexes:

                #creates list of NA's and appends it to merge_table
                na_list = []
                for k in range(len(new_headers)):
                    na_list.append("NA")
                merge_table.append(na_list)
                merge_table_index+=1

                # loops through current instance with no match and adds values to current NA list
                for j in range(len(other_table.data[i])):
                    for l in range(len(new_headers)):
                        if (new_headers[l] == other_table.column_names[j]):
                            merge_table[merge_table_index][l] = other_table.data[i][j]

        #creates new table and returns it
        newTable = MyPyTable(data=merge_table, column_names=new_headers)
        return newTable # TODO: fix this