import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Metrica de similitud por coseno
def compute_similarity(example, query_embedding):
    embedding = example['embeddings'] 
    similarity = util.cos_sim(embedding, query_embedding).item()
    return similarity


def main():
    buscar = 'y'
    while buscar == 'y':
        #sentences = []
        df = pd.read_csv('./resources/IMDB top 1000.csv')
        print(df.head())

        query = input('Ingresa el termino de busqueda: ')
        #sentences.append(query)

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(df['Description'],batch_size=64,show_progress_bar=True)
        df['embeddings'] = embeddings.tolist()

        #query_embedding = model.encode(sentences)[0]
        query_embedding = model.encode([query])[0]
        df['similarity'] = df.apply(lambda x: compute_similarity(x, query_embedding), axis=1)

        # Aumentar contexto --> "Certificate" + "Genre"
        df['Classification'] = df['Certificate'].astype(str) + ' | ' + df['Genre'].astype(str)

        df = df.sort_values(by='similarity', ascending=False)        

        # Elimina resultados repetidos
        print(df.head()[['Title','Classification']].drop_duplicates())

        buscar = input('Desea realizar una nueva busqueda? y/n: ')


if __name__ == '__main__':
    main()