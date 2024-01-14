from gensim.summarization import bm25
import pickle
import jieba
import re
import os
import pandas as pd

# 加载数据集
# search_fold = "../Data/search_data/"
# filelist = os.listdir(search_fold)
# frames = []
# for f in filelist:
#     file = search_fold + f
#     df = pd.read_csv(file)
#     frames.append(df)
# search_df = pd.concat(frames)

# search_df = pd.read_csv("../Data/search_product_full.csv")
# print(search_df.shape)
# search_df.drop_duplicates(inplace=True)
# print(search_df.shape)
# search_df.drop_duplicates(subset=['sku_id'], inplace=True)
# print(search_df.shape)
#
# # Load segmentation dictionary
# with open("../Data/temp/content_dict.pkl", "rb") as f:
#     content_dict = pickle.load(f)
# print(len(content_dict))
#
# # Build the bm25 model
# corpus = content_dict.values()
# bm25Model = bm25.BM25(corpus)
# with open("../Data/temp/bm25Model.pkl", "wb") as f:
#     pickle.dump(bm25Model, f)
#
# # Load the bm25 model
# with open("../Data/temp/bm25Model.pkl", "rb") as f:
#     bm25Model = pickle.load(f)


class Searcher:
    def __init__(self):
        with open("static/Data/temp/content_dict.pkl", "rb") as f:
            content_dict = pickle.load(f)
        self._content_dict = content_dict   # Segmentation list of product titles
        with open("static/Data/temp/bm25Model.pkl", "rb") as f:
            bm25Model = pickle.load(f)      # bm25 model
        self._bm25 = bm25Model
        self._search_df = pd.read_csv("static/Data/search_product_full.csv")  # Search data frame
        with open("static/Data/temp/sales_tags.pkl", "rb") as f:
            sales_tag = pickle.load(f)      # Sales tag running result
        self._sales_tag = sales_tag
        self._cluster_df = pd.read_csv("static/Data/temp/cluster_infos.csv")
        with open("static/Data/temp/cluster_center.pkl", "rb") as f:  # Cluster coordinates
            cluster_center = pickle.load(f)
        self._cluster_center = cluster_center  # Cluster center
        self._radar_df = pd.read_csv("static/Data/temp/rank_data.csv")   # Radar chart

    @staticmethod
    def get_key(d, value):
        """
        Return the key of the tokens
        :param d: content_dict
        :param value: a list with tokens
        :return: the key of the tokens
        """
        res = [k for k, v in d.items() if v == value]
        return res[0]

    @staticmethod
    def recommend(res: dict, has_query, num=10) -> list:
        """
        Perform word frequency statistics on search results and select high-frequency words as search recommendations for user clicks
        :param res: Matched search results (dictionary, key is the matching score, value is the token result)
        :param has_query: Words already appearing in the query
        :param num: Number of recommended words
        :return: Recommended word list
        """
        word_freq = dict()
        for score, tokens in res.items():
            for token in tokens:
                if token not in word_freq.keys():
                    word_freq[token] = 1
                else:
                    word_freq[token] += 1
        # Sorting
        word_freq = dict(sorted(word_freq.items(),
                                key=lambda x: x[1], reverse=True))
        # Selecting top words
        recommend_word_list = []
        i = 1
        for word, cnt in word_freq.items():
            if word != " " and word not in has_query:
                i += 1
                recommend_word_list.append(word)
                if i > num:
                    break
        return recommend_word_list

    def get_title(self, sku_id):
        """
        Retrieve the title from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Title
        """
        res = self._search_df[self._search_df['sku_id'] == sku_id]['title'].values[0]
        if res != res:
            res = 'No data available'
        return res

    def get_second_title(self, sku_id):
        """
        Retrieve the second title from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Second title
        """
        res = self._search_df[self._search_df['sku_id'] == sku_id]['second_title'].values[0]
        if res != res:
            res = 'No data available'
        return res

    def get_price(self, sku_id):
        """
        Retrieve the price from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Price
        """
        res = self._search_df[self._search_df['sku_id'] == sku_id]['price'].values[0]
        if res != res:
            res = 'No data available'
        return res

    def get_reputation(self, sku_id):
        """
        Retrieve the reputation (with possible null values) from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Reputation (with possible null values)
        """
        res = self._search_df[self._search_df['sku_id'] == sku_id]['reputation'].values[0]
        if res != res:
            res = 'No data available'
        return res

    def get_comment_num(self, sku_id):
        """
        Retrieve the recent comment count from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Recent comment count
        """
        res = self._search_df[self._search_df['sku_id'] == sku_id]['comment_num'].values[0]
        if res != res:
            res = 'No data available'
        return res

    def get_img_link(self, sku_id):
        """
        Retrieve the image link from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Image link of the product
        """
        res = self._search_df[self._search_df['sku_id'] == sku_id]['img_link'].values[0]
        return res

    @staticmethod
    def get_img_path(sku_id):
        """
        Find the location of the image in the 'static' folder using the product's sku_id
        :param sku_id: Product's sku_id
        :return: If found, return the local path; if not found, return an empty string
        """
        folder = "static/Data/item_pic/"
        img_list = os.listdir(folder)
        if "{:}.jpg".format(sku_id) in img_list:
            res = '/' + folder + "{:}.jpg".format(sku_id)
        else:
            res = ""
        return res

    def get_word_cloud(sku_id):
        """
        Find the location of the word cloud image in the 'static' folder using the product's sku_id
        :param sku_id: Product's sku_id
        :return: If found, return the local path; if not found, return an empty string
        """
        folder = "static/Data/plot/word_cloud/"
        img_list = os.listdir(folder)
        if "{:}.png".format(sku_id) in img_list:
            res = '/' + folder + "{:}.png".format(sku_id)
        else:
            res = ""
        return res

    @staticmethod
    def get_word_freq(sku_id):
        """
        Find the location of the word frequency image in the 'static' folder using the product's sku_id
        :param sku_id: Product's sku_id
        :return: If found, return the local path; if not found, return an empty string
        """
        folder = "static/Data/plot/word_freq/"
        img_list = os.listdir(folder)
        if "{:}.jpg".format(sku_id) in img_list:
            res = '/' + folder + "{:}.jpg".format(sku_id)
        else:
            res = ""
        return res

    def get_special(self, sku_id):
        """
        Retrieve the tag label list for the product from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Tag label list for the product
        """
        res = self._search_df[self._search_df['sku_id'] == sku_id]['special'].values[0]
        return eval(res)

    def get_sales(self, sku_id):
        """
        Retrieve marketing labels from the label result set using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Product marketing label list
        """
        return self._sales_tag[sku_id]

    @staticmethod
    def df2list_coordinate(df):
        """
        Convert data frame of the same category into a two-dimensional list
        :param df: Data frame
        :return: Two-dimensional list
        """
        res = []
        for ind, row in df.iterrows():
            temp = list([row['x'], row['y'], 5, row['content'], row['label']])
            res.append(temp)
        return res

    def get_coordinate(self, sku_id):
        """
        Retrieve labels for the product from the data table using the product's sku_id
        :param sku_id: Product's sku_id
        :return: Three-dimensional list, containing three two-dimensional lists, each comment corresponding to a list [x, y, size, text, category]
        """
        res = []
        sku_comment_df = self._cluster_df[self._cluster_df['sku_id'] == sku_id]
        sku_comment_df_0 = sku_comment_df[sku_comment_df['label'] == 0]
        x0, y0 = sku_comment_df_0.describe()['x'].mean(), sku_comment_df_0.describe()['y'].mean()
        sku_comment_df_1 = sku_comment_df[sku_comment_df['label'] == 1]
        x1, y1 = sku_comment_df_1.describe()['x'].mean(), sku_comment_df_1.describe()['y'].mean()
        sku_comment_df_2 = sku_comment_df[sku_comment_df['label'] == 2]
        x2, y2 = sku_comment_df_2.describe()['x'].mean(), sku_comment_df_0.describe()['y'].mean()
        res.append(self.df2list_coordinate(sku_comment_df_0))
        res.append(self.df2list_coordinate(sku_comment_df_1))
        res.append(self.df2list_coordinate(sku_comment_df_2))
        res.append([[x0, y0, 10, '0 class cluster center', 3],
                    [x1, y1, 10, '1 class cluster center', 3],
                    [x2, y2, 10, '2 class cluster center', 3]])
        return res

    def get_center(self, sku_id):
        """
        Return the corresponding cluster center based on the product's sku_id
        :param sku_id: Product's sku_id
        :return: List of cluster centers
        """
        if sku_id in self._cluster_center.keys():
            res = self._cluster_center[sku_id]
            res = res.tolist()
            for i in range(len(res)):
                res[i].extend([10, " ", 3])  # Append size, text, and category after each cluster center coordinate
        else:
            res = [[]]
        return res

    def get_radar(self, sku_id):
        """
        Return the coordinates needed for the radar chart based on the product's sku_id
        :param sku_id: Product's sku_id
        :return: Two-dimensional list, containing one sample point in the outer layer and the inner layer
        """
        sku_radar = self._radar_df[self._radar_df['sku'] == sku_id]
        sku_radar = sku_radar[["comment_num_rank", "price_rank",
                               "reputation_rank"]].values.tolist()
        res = sku_radar[0]
        return res

    def tag_filter(self, res_dict, tag):
        """
        Filter products that meet the criteria using tag information
        :param res_dict: Dictionary obtained from the search (key: score, value: tokenized result)
        :param tag: Tag list
        :return: Filtered search dictionary
        """
        # Tag filtering
        scores_dict = dict()
        for score, value in res_dict.items():
            key = self.get_key(self._content_dict, value)
            tag_list = self.get_special(key)
            for t in tag:  # Iterate over selected tag options
                if t == 1:  # Free shipping
                    if 'Free Shipping' in tag_list:
                        scores_dict[score] = value
                elif t == 2:  # New arrival
                    if 'New Arrival' in tag_list:
                        scores_dict[score] = value
                elif t == 3:  # Discount
                    for c in tag_list:  # Iterate over each tag to check if it contains a number
                        if len(re.findall(r'\d', c)) > 0:
                            scores_dict[score] = value
                            break
                elif t == 4:  # JD Logistics
                    if 'JD Logistics' in tag_list:
                        scores_dict[score] = value
                else:
                    scores_dict = res_dict
        return scores_dict

    def price_filter(self, res_dict, min_price, max_price):
        """
        Filter products based on the price range
        :param res_dict: Dictionary obtained from the search (key: score, value: tokenized result)
        :param min_price: Minimum price, default is 0
        :param max_price: Maximum price, default is 99999
        :return: Filtered search dictionary
        """
        scores_dict = dict()
        for score, value in res_dict.items():
            key = self.get_key(self._content_dict, value)
            price = self.get_price(key)
            if min_price <= price <= max_price:
                scores_dict[score] = value
        return scores_dict

    def price_sort(self, res_dict, reverse=0) -> list:
        """
        Sort by price
        :param res_dict: Dictionary obtained from the search (key: score, value: tokenized result)
        :param reverse: Whether to output in reverse order, default is ascending order by price, if specified as True, then descending order
        :return: Return the sku_id sequence
        """
        price_dict = dict()
        for score, value in res_dict.items():
            key = self.get_key(self._content_dict, value)
            if self.get_price(key) != 'No Data':
                price_dict[key] = self.get_price(key)
        price_dict = dict(sorted(price_dict.items(),
                                 key=lambda x: x[1], reverse=reverse))
        return list(price_dict.keys())

    def reputation_sort(self, res_dict, reverse=1):
        """
        Sort by reputation rate
        :param res_dict: Dictionary obtained from the search (key: score, value: tokenized result)
        :param reverse: Whether to output in reverse order, default is descending order by reputation rate, if specified as False, then ascending order
        :return: Return the sku_id sequence
        """
        reputation_dict = dict()
        for score, value in res_dict.items():
            key = self.get_key(self._content_dict, value)
            if self.get_reputation(key) != 'No Data':  # Reputation might be empty
                reputation_dict[key] = self.get_reputation(key)
            else:
                reputation_dict[key] = 0
        reputation_dict = dict(sorted(reputation_dict.items(),
                                      key=lambda x: x[1], reverse=reverse))
        return list(reputation_dict.keys())

    def comment_num_sort(self, res_dict, reverse=1):
        """
        Sort by sales volume, using comment number as a substitute
        :param res_dict:
        :param reverse:
        :return:
        """
        # First clean up the comment_num to make it numeric
        temp_df = self._search_df.copy()  # Copy of the data frame, modifications won't affect the original data
        for ind, row in temp_df.iterrows():
            comment_num = row['comment_num']
            if comment_num == comment_num and comment_num[-1] == '+':
                comment_num = comment_num[:-1]
                if '万' in comment_num:
                    comment_num = int(comment_num[:-1]) * 10000
                else:
                    comment_num = int(comment_num)
            else:
                comment_num = int(comment_num)  # Convert other strings to numeric
            temp_df.loc[ind, 'comment_num'] = comment_num

        # Perform sales volume sorting (needs to be based on the cleaned data frame)
        quantity_dict = dict()
        for score, value in res_dict.items():
            key = self.get_key(self._content_dict, value)
            if self.get_comment_num(key) != 'No Data':
                # Use the temporary data frame to get the numeric form of the comment quantity
                res = temp_df[temp_df['sku_id'] == key]['comment_num'].values[0]
                quantity_dict[key] = res

        quantity_dict = dict(sorted(quantity_dict.items(),
                                    key=lambda x: x[1], reverse=reverse))
        return list(quantity_dict.keys())

    def search(self, search_text: str, recommend_list: list,
               tag: list, sort_method: int, reverse: int,
               min_price=0, max_price=99999):
        """
        Main search function
        :param search_text: Search text
        :param recommend_list: Search keywords
        :param tag: Tag filtering (free shipping, discount, new arrival)
        :param sort_method: Sorting method (default by relevance 0, price sorting 1, reputation rate sorting 2, sales sorting 3)
        :param reverse: Sorting direction (0 ascending, 1 descending)
        :param min_price: Minimum price
        :param max_price: Maximum price
        :return: Tuple containing SKU list and recommended keywords
        """
        print("Start searching...")
        # Build corpus and match using BM25 algorithm
        corpus = self._content_dict.values()
        test = jieba.lcut(search_text)
        if len(recommend_list) > 0:
            test = recommend_list
        scores = self._bm25.get_scores(test)
        scores_dict = dict(zip(scores, corpus))
        scores_dict = dict(sorted(scores_dict.items(),
                                  key=lambda x: x[0], reverse=True))

        # Use tag filtering
        scores_dict = self.tag_filter(scores_dict, tag)

        # Use price filtering
        scores_dict = self.price_filter(scores_dict, min_price, max_price)

        # Select the top 200 samples
        scores_dict = self.top_dict(scores_dict, max_num=100)

        # Specify the sorting method
        if sort_method == 1:
            sku_list = self.price_sort(scores_dict, reverse=reverse)
        elif sort_method == 2:
            sku_list = self.reputation_sort(scores_dict, reverse=reverse)
        elif sort_method == 3:
            sku_list = self.comment_num_sort(scores_dict, reverse=reverse)
        else:
            quantity_dict = dict()
            for score, value in scores_dict.items():
                key = self.get_key(self._content_dict, value)
                quantity_dict[key] = value
            sku_list = list(quantity_dict.keys())
        # Recommend 5 keywords
        recommend_list = self.recommend(scores_dict, test, 5)
        return sku_list, recommend_list

    def run(self, search_text: str, recommend_list: list,
            tag: list, order: str, min_price="0", max_price="99999") -> list:
        """
        Main function for running, used to interface with the frontend adjustment interface
        :param search_text: Search by text
        :param recommend_list: Search by keywords
        :param tag: Tag filtering (1 free shipping, 2 discount, 3 new arrival, 4 JD Logistics)
        :param order: 1 price ascending, 2 price descending, 3 reputation ascending, 4 reputation descending, 5 sales ascending, 6 sales descending
        :param min_price: Minimum price
        :param max_price: Maximum price
        :return: List containing information for all products, each product stored in a dictionary with the following format:
                {"sku_id": "xxx", "name": "xxx", "subname": "xxx", "imgpath": "xxx",
                 "salestag": ["a", "b", "c"], "fixtags": ["a", "b", ...],
                 "commentrate": "xxx", "commentnum": "xxx"}
        """
        # Adapt tag filtering method
        tag = [int(t) for t in tag]

        # Adapt sorting method
        if order == "1":
            sort_method = 1;
            reverse = 0
        elif order == "2":
            sort_method = 1;
            reverse = 1
        elif order == "3":
            sort_method = 2;
            reverse = 0
        elif order == "4":
            sort_method = 2;
            reverse = 1
        elif order == "5":
            sort_method = 3;
            reverse = 0
        elif order == "6":
            sort_method = 3;
            reverse = 1
        else:
            sort_method = 0;
            reverse = 1

        # Adapt min_price and max_price
        min_price = int(min_price)
        max_price = int(max_price)

        # Call the main search function
        search_list, recommend_list \
            = self.search(search_text=search_text,
                          recommend_list=recommend_list,
                          tag=tag,
                          sort_method=sort_method,
                          reverse=reverse,
                          min_price=min_price,
                          max_price=max_price
                          )
        result = []  # List to store product information
        for k in search_list:
            try:
                temp_dict = dict()  # Create a dictionary to store all information for the product
                temp_dict["sku_id"] = str(k)
                temp_dict["name"] = self.get_title(k)
                temp_dict["subname"] = self.get_second_title(k)
                temp_dict["imgpath"] = self.get_img_path(k)
                temp_dict["salestag"] = self.get_sales(k)
                temp_dict["fixtags"] = self.get_special(k)
                temp_dict["price"] = str(self.get_price(k))
                if self.get_reputation(k) != 'No Data':
                    temp_dict["commentrate"] = str(int(self.get_reputation(k))) + "%"
                else:
                    temp_dict["commentrate"] = str(self.get_reputation(k))
                temp_dict["commentnum"] = str(self.get_comment_num(k))
                temp_dict["coordinate"] = self.get_coordinate(k)
                temp_dict["radar"] = self.get_radar(k)
                result.append(temp_dict)
            except IndexError:
                pass
        return result, recommend_list

if __name__ == '__main__':
    s = Searcher()
    # search_list = s.search(search_text="皮尔卡丹男士夹克",

    # search_list = s.search(search_text="皮尔卡丹男士夹克",
    #                        recommend_list=[],
    #                        tag=[2, 1],     # '新品', '免邮'
    #                        sort_method=2,  # 好评率排序
    #                        reverse=1,
    #                        min_price=100,
    #                        max_price=5000
    #                       )
    #
    # print(len(search_list))
    # for k in search_list:
    #     print("-------{:}------".format(k))
    #     print(s.get_title(k))
    #     print(s.get_second_title(k))
    #     print(s.get_price(k), s.get_reputation(k), s.get_comment_num(k))
    #     print(s.get_special(k))
    #     print(s.get_sales(k))
    #     print(s.get_coordinate(k))
    #     print(s.get_center(k))

    goodslist, recommend_list = \
        s.run(search_text="皮尔卡丹男士风衣",
              recommend_list=[],  #
              tag=["2", "4"],
              order="0",
              min_price="100",
              max_price="1000"
              )
    print(goodslist)
    print(recommend_list)
    # with open("test.pkl", "wb") as f:
    #     pickle.dump(goodslist, f)

