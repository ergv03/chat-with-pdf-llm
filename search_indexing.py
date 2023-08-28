from langchain import FAISS
from langchain.document_loaders import PyPDFium2Loader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import pypdfium2 as pdfium
from constants import chunk_size, chunk_overlap, number_snippets_to_retrieve
from langchain.document_loaders import WebBaseLoader
import re
from requests.utils import requote_uri


def download_and_index_pdf(urls: list[str]) -> FAISS:
    """
    Download and index a list of PDFs based on the URLs
    """

    def __split_text(text):
        """
        Helper function. In order to create the URLs with "Scroll to Text Fragment" feature, split a text using regex,
        and encode the substrings
        """

        split_text = re.split('\n|,', text)
        split_text = [re.sub('\s+$|^\s+', '', s) for s in split_text]
        split_text = [requote_uri(s) for s in split_text]
        split_text = [s.replace('-', '%2D') for s in split_text]

        return split_text

    def __update_metadata(pages, url, doc_type):
        """
        Add to the document metadata the title and original URL
        """
        for page in pages:
            page.metadata['doc_type'] = doc_type
            if doc_type == 'pdf':
                # For snippets extracted from PDFs, set the Document's metadata title the same as the original PDF
                pdf = pdfium.PdfDocument(page.metadata['source'])
                title = pdf.get_metadata_dict().get('Title', url)
                page.metadata['source'] = url
                page.metadata['title'] = title
            else:
                # For snippets extracted from a HTML document, define as source a URL to the original HTML doc with the 
                # respective snippet highlighted as quote ("Scroll to Text Fragment")
                # References: https://chromestatus.com/feature/4733392803332096
                # https://stackoverflow.com/questions/62161819/what-exactly-is-the-text-location-hash-in-an-url
                split_text = __split_text(page.page_content)
                source = f"{page.metadata['source']}#:~:text={split_text[0]},{split_text[-1]}"
                page.metadata['source'] = source
        
        return pages

    all_pages = []
    for url in urls:
        doc_type = 'pdf' if '.pdf' in url else 'html'
        if doc_type == 'pdf':
            loader = PyPDFium2Loader(url)
        else:
            loader = WebBaseLoader(url)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pages = loader.load_and_split(splitter)        
        pages = __update_metadata(pages, url, doc_type)
        all_pages += pages

    faiss_index = FAISS.from_documents(all_pages, OpenAIEmbeddings())

    return faiss_index


def search_faiss_index(faiss_index: FAISS, query: str, top_k: int = number_snippets_to_retrieve) -> list:
    """
    Search a FAISS index, using the passed query
    """

    docs = faiss_index.similarity_search(query, k=top_k)

    return docs
