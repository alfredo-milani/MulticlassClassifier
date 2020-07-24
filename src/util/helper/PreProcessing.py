class PreProcessing(object):
    """
    Contains helper methods
    """

    @staticmethod
    def get_na_count(dataset) -> int:
        """
        Function counting missing values for each feature from dataset
        :param dataset: dataset
        :return: sum of missing values
        """
        # per ogni elemento (i,j) del dataset, isna() restituisce
        # TRUE/FALSE se il valore corrispondente è mancante/presente
        boolean_mask = dataset.isna()
        # contiamo il numero di TRUE per ogni attributo sul dataset
        return boolean_mask.sum(axis=0)
