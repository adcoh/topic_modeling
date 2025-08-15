from pathlib import Path


class DataReader:
    """A class for reading data files and converting them to document format."""
    
    def __init__(self, data_dir: str | Path = "data"):
        """
        Initialize the DataReader with a data directory.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
    
    def read_txt_file(self, file_path: str | Path) -> list[str]:
        """
        Read a text file and split it into documents.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of strings, where each string is a document
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines or other document separators
        # This can be customized based on the specific format
        documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        
        # If no double newlines found, treat each line as a document
        if len(documents) == 1 and '\n' in content:
            documents = [line.strip() for line in content.split('\n') if line.strip()]
        
        return documents
    
    def read_arxiv_corpus(self, filename: str = "arxiv_corpus.txt") -> list[str]:
        """
        Read the arxiv corpus file specifically.
        
        Args:
            filename: Name of the arxiv corpus file
            
        Returns:
            List of strings, where each string is a document
        """
        file_path = self.data_dir / filename
        return self.read_txt_file(file_path)
    
    def read_all_txt_files(self) -> list[str]:
        """
        Read all .txt files in the data directory and combine them.
        
        Returns:
            List of strings, where each string is a document
        """
        all_documents = []
        
        for txt_file in self.data_dir.glob("*.txt"):
            try:
                documents = self.read_txt_file(txt_file)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Could not read {txt_file}: {e}")
        
        return all_documents
    
    def list_available_files(self) -> list[str]:
        """
        List all available data files in the data directory.
        
        Returns:
            List of filenames
        """
        return [f.name for f in self.data_dir.iterdir() if f.is_file()]
    
    def get_document_count(self, filename: str | None = None) -> int:
        """
        Get the number of documents in a specific file or all files.
        
        Args:
            filename: Specific filename to count, or None for all files
            
        Returns:
            Number of documents
        """
        if filename:
            documents = self.read_txt_file(self.data_dir / filename)
            return len(documents)
        else:
            documents = self.read_all_txt_files()
            return len(documents)
