import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel, LongformerTokenizer

pretrained_model = '../hub/swift/longformer-base-4096'

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=4, trans_layers=2, dim_feedforward=512):
        super(TransformerEncoderClassifier, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward),
            num_layers=trans_layers
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)  # (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)  # 平均池化获取全局表示
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.longformer = LongformerModel.from_pretrained(pretrained_model)
        for param in self.longformer.parameters():
            param.requires_grad = False  # 冻结 Longformer 参数
        inchannel = self.longformer.config.hidden_size
        self.transformer_encoder = TransformerEncoderClassifier(inchannel, 2)

    def forward(self, x):
        x = self.longformer(x)[0]  # (batch_size, seq_len, hidden_dim)
        x = self.transformer_encoder(x)
        return x



text = "r'Although the streets of South Philly do not have the glamour typically associated with Rome, the ethnic neighborhood definitely boasts some of the best Italian food in America. Since I am always on the hunt to find great Italian food (as it is my favorite kind of food to eat), I was very excited when my mom told me that we would be trying Modo Mio on Girard Ave for a family dinner. Our waiter suggested doing the \"La Turista Menu,\" which is a four-course meal composed of an appetizer, pasta, entr篓娄e, and dessert. And for the price of just $33, none of us could resist! The best part was that it was not a pri fix or limited menu. You got to order off of the regular menu and select one item of your choosing from each category. But before we even had the opportunity to order, food was already being brought to the table. We received a plate of complementary Bruschetta Toasted topped with Grilled Eggplant and Cinnamon as well as Fresh Grilled Ciabatta Bread with Homemade Ricotta Cheese and Olive Oil infused with Porcini Mushrooms. My mom always says that you can judge a restaurant by the quality of its bread, and you better believe that we all got pretty excited for our entrees after trying this stuff. It was amazing! The ricotta was as creamy and smooth as butter and the olive oil had a tremendous earthy flavor from the mushrooms. The bread was also fantastic with a nice crispy crust and a soft warm center. Now that our appetites were going, we got to ordering our courses. For my appetizer I had the Creamy Potatoes with Lump Crab and Sweet Roasted Red Peppers. I was unsure as to how it was going to be served, and I was pleasantly surprised when it came out on top of more of that delicious bread!! It was definitely a rich appetizer, but it was too good to leave any behind. The crabmeat was abundant and not overshadowed by the potatoes. The roasted red peppers were perfectly placed atop the whole dish, and they added a great sweetness to every bite. I also tried the Mussels in Spicy Red Sauce, which were so tender and the best that I have ever had out! The sauce was great with salty capers and just the right amount of red pepper flakes for heat. My sister ordered the Egg Dipped and Fried Mozzarella Grill Cheese Sandwich with Lemon Caper Sauce, which I thought would be the grand-daddy of all grill cheeses, but I was disappointed by the strength of the lemon sauce. I thought it was too pungent and overwhelmed what could have been a really great dish. My other sister, Hannah, ordered the Calamari Stewed in Red Sauce with Capers and Olives. I thought it was good, but still prefer crispy fried calamari (call me unrefined...but at the end of the day they taste better to me!). I also had a bite of my dad's Seared Scallop atop Fried Chickpea Pancake. The chickpea pancake was fabulous and so was the scallop, but the portion was rather small for my liking. My favorite course of the meal was the pasta! The next time I go to Modo Mio, I am going to ask to order two pasta dishes instead of an entr篓娄e. I ordered the Buccatini Amatriciana, which is a thick homemade spaghetti with pancetta in a spicy red sauce. The pasta was actual perfection, from texture to cook time, and the sauce was very nice. It was rich (and a bit oily) because of the pancetta, but worth every calorie! The Gnocchi with Porcini Mushrooms, Peas, and Pancetta in Gorgonzola Sauce was also rich, but undoubtedly the best gnocchi I have ever had the privilege of tasting. They were soft delicate pillows of potato and pasta that melted in your mouth. I was also pleasantly surprised by the gorgonzola cream sauce, which in some miraculous way tasted light! My other favorite pasta dish was the Carmelli Pasta Purses filled with Taleggio Cheese in a truffle-rosemary butter sauce. This dish was probably the richest of all the pasta dishes, but also really, really, good. Taleggio cheese is very strong (sometimes called the \"stinky cheese\") but when it is melted and gooey, it is delicious. My least favorite pasta was the Lasagna topped with a Fried Egg. I thought it was too rich to even taste good. The plate was covered in oil and the fried egg was just too much. I only had one bite and found that I couldn't get the greasy feeling out of my mouth. For my entr篓娄e I ordered Skirt Steak with Green Olives and Fried Chickpeas. Although it was cooked perfectly, and tender and juicy, I thought it was way too heavy after all the food that I had previously eaten. I enjoyed the finely chopped green olives atop the steak, but the fried chickpeas, which were done in little flattened strips were far too oily. I would definitely recommend ordering a fish as an entr篓娄e, because it is on the relatively lighter side. My dad ordered the winning entr篓娄e, which was Grilled Swordfish with balsamic caramelized onion, golden raisins, pine nuts radicchio, smoked mozzarella, and rosemary pangrattugiato. This flavor packed entr篓娄e, just seems to melt in your mouth. It was by far my favorite entr篓娄e."
tokenizer = LongformerTokenizer.from_pretrained(pretrained_model)
tokens = tokenizer(text, add_special_tokens=True, return_tensors='pt')
# add_special_tokens=True表示在文本两端添加特殊标记（如[CLS]和[SEP]）
# return_tensors='pt'表示返回的结果以 PyTorch 张量的形式呈现
input_ids = tokens['input_ids']
print("输入张量:", input_ids.shape)
print(input_ids)
model = Model()
x = model.forward(input_ids)
print(x.size())