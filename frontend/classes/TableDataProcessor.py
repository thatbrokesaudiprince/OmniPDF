class TableDataProcessor:
    """
    A class for processing table data to make it suitable for JSON serialization.
    """

    @staticmethod
    def format_for_json(table: list) -> list:
        """
        Formats table data for JSON serialization, handling special cases.

        Args:
            table: A nested list representing the table data.

        Returns:
            A processed nested list suitable for JSON conversion.
        """
        processed_table = []
        for row in table:
            processed_row = []
            for cell in row:
                if isinstance(cell, str):
                    # Replace newlines and single quotes, and handle "null" string.
                    processed_cell = cell.replace('\n', '\\n').replace("'", '"')
                    processed_cell = None if processed_cell == "null" else processed_cell
                else:
                    processed_cell = None if cell is None else cell

                processed_row.append(processed_cell)
            processed_table.append(processed_row)
        return processed_table
