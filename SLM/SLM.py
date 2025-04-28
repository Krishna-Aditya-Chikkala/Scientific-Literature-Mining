import os
import subprocess
from features.arxiv_client import search_arxiv
from features.trend_analysis import perform_trend_analysis

def main():
    print("Welcome to Scientific Literature Mining!")
    
    text_corpus = None  # Store corpus for trend analysis

    while True:
        print("\nChoose an option:")
        print("1. Search for Documents on arXiv")
        print("2. Perform Trend Analysis")
        print("3. Exit")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
            query = input("Enter your query: ").strip()
            num_results = int(input("Enter the number of results you want: ").strip())
            print("\nSearching for documents...\n")
            try:
                top_documents, text_corpus = search_arxiv(query, num_results)
                print("\nTop Documents Found:")
                for doc in top_documents:
                    print(f"Title: {doc['title']}")
                    print(f"Authors: {doc['authors']}")
                    print(f"Abstract: {doc['abstract']}")
                    print(f"PDF URL: {doc['pdf_url']}\n")
            except Exception as e:
                print(f"Error occurred during search: {e}")

        elif choice == '2':
            if text_corpus:
                print("\nRunning Trend Analysis on Retrieved Documents...\n")
                try:
                    perform_trend_analysis(text_corpus)  # Pass text_corpus
                except Exception as e:
                    print(f"Error during Trend Analysis: {e}")
            else:
                print("\nNo documents found! Please perform a search first.\n")

        elif choice == '3':
            print("Exiting the program...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == '__main__':
    main()
