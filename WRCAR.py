#import and df load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
df = pd.read_csv(r'C:\Users\Preston Conner\Desktop\WRCAR\World Relief Content Analysis Research Report.csv')

df
df.nunique()
df.isnull().sum()
df.columns

#main varibles of interest is engagement (likes and commments)

#create a pie chart of the different nonprofits and ammount of posts analyzed
df_nonprofit_counts = df['nonprofit'].value_counts()
plt.figure(figsize=(8, 6))
labels = df_nonprofit_counts.index
counts = df_nonprofit_counts.values
explosion = [0., 0., 0.1, 0.]
total_posts = sum(counts)
#turn abbreviation names to full names
abbreviation_to_full = {'WR': 'World Relief', 'DR': 'Direct Relief', 'IRC': 'International Rescue Committee',
                        'CI': 'Compassion International'}
labels = [abbreviation_to_full[label] for label in labels]
#custom respective colors for each nonprofit
custom_colors = ['#1d62d3', '#21beec', '#fbc325', '#fc6c2c']
plt.pie(counts, labels=labels, autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * total_posts / 100, p), 
        startangle=90, colors=custom_colors, explode=explosion, shadow=True, wedgeprops={'edgecolor': 'grey'})
plt.title('Number of Analyzed Posts by Nonprofit', fontsize=16)
plt.axis('equal')
plt.savefig('Pie Chart (Posts Analyzed)', bbox_inches='tight')
plt.show


#z score to check for outliers
#will be ommitted from set but will not be ignored, more in report
z_scores = np.abs((df['likes'] - df['likes'].mean())/ df['likes'].std())
z_score_threshold = 2
z_score_outliers = df[z_scores > z_score_threshold]
print(z_score_outliers)
num_like_outliers = z_score_outliers.shape[0]
print("Outliers: " + str(num_like_outliers))
#four outliers
#dataframe without outliers
df_without_like_outliers = df[~(z_scores > z_score_threshold)]
print(df_without_like_outliers)


#find distribution of posts by months and days
#first have to convert date column to datetime format
df_without_like_outliers['date'] = pd.to_datetime(df_without_like_outliers['date'], format='%d-%b')
#extract month from date column
df_without_like_outliers['month'] = df_without_like_outliers['date'].dt.month

#distribution of posts across months
#infer wont be of much use, more interested on days
month_map = {4: 'April', 5: 'May', 6: 'June'}
df_without_like_outliers['month'] = df_without_like_outliers['month'].map(month_map)
#count occurrences of each month
month_counts = df_without_like_outliers['month'].value_counts()
month_counts = month_counts.sort_index()
#make plot of posts analyzed across months
plt.figure(figsize=(8, 6))
plt.bar(month_counts.index, month_counts.values)
#places value on top of bars
for x, y in zip(month_counts.index, month_counts.values):
    plt.text(x, y, str(y), ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=14)
plt.xlabel('Month', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Distribution of Posts Analyzed across Months', fontsize=16)
plt.savefig('Distribution of Posts Analyzed Across Months')
plt.show()

#plot distribution of posts by days
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.countplot(x='day', data=df_without_like_outliers, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xticks(fontsize=12)
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Distribution of Post by Day Posted (All Nonprofits)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Distribution of Posts Analyzed by Day Posted (All Nonprofits)')
plt.show()


#create a correlation matrix heatmap of all the variables to find initial overall trends
#this is to just get a general idea/understand data, more interested on trends for each nonprofit
sns.set(style='whitegrid')
plt.figure(figsize=(16,12))
#dropping ID, nonprofit, date, day from heatmap to include only quantitative data
#will turn days into boolean afterwards
df_without_nonprofit_id = df_without_like_outliers.drop(['nonprofit', 'ID', 'date', 'day', 'month'], axis=1)
sns.heatmap(df_without_nonprofit_id.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.xticks(rotation=90, fontsize=17)
plt.yticks(rotation=0, fontsize=17)
plt.title('Correlation Matrix Heatmap (All 4 Nonprofits)', fontsize=25)
plt.savefig('Correlation Matrix Heatmap (All Nonprofits with 4 outliers ommitted)')
plt.show
#initial overall correlations to note:
# 1) likes/ind_quotes 2) likes/comments;. 3) likes/post_types & likes/emojis 4) likes/partnership
# 5) likes/graphic 6) likes/org_persons 7) comments/emojis 8) comments/holiday 9) hashtags/graphics
# 10) graphic/graphic_image 11)graphic_image/per_story 12)graphic_image/persons_in_need
# 13)graphic_image/org_persons 14) short_form_vid/persons_in_need 15)emojis/likes
# 16) emojies/likes 17)per_story/persons_in_need 18)per_story/graphic_image 19)per_story/post_type

#using pd.get_dummies to turn Days into boolean to seperate dataframes
df_dummies = pd.get_dummies(df_without_like_outliers, columns=['day'])
#make sure changes took place
df_dummies.dtypes
#no need to drop days because dummies does already
df_dummies_without_nonprofit_ID_date= df_dummies.drop(['nonprofit', 'ID', 'date', 'month'], axis=1)
#print correlation matrix
corr_matrix = df_dummies_without_nonprofit_ID_date.corr()
like_corr = corr_matrix['likes']
comment_corr = corr_matrix['comments']
print(like_corr)
print(comment_corr)

#make heatmap including boolean days
sns.set(style='whitegrid')
plt.figure(figsize=(24,20))
#dont have to do drop value because did earlier with dummies
df_heatmap_boolean_days = df_dummies_without_nonprofit_ID_date
sns.heatmap(df_heatmap_boolean_days.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.xticks(rotation=90, fontsize=17)
plt.yticks(rotation=0, fontsize=17)
plt.title('Correlation Matrix Heatmap (All 4 Nonprofits); boolean days added', fontsize=25)
plt.show
#did not give any insights
#potential leads could just be marked up coincidences because the time frame we chose
#for example holidays being a postive correlation with tuesdays

#start looking at indiv trends
#seperate dataframe into each nonprofit by unique values in nonprofit column
nonprofit_groups = df_without_like_outliers.groupby('nonprofit')
df_irc = nonprofit_groups.get_group('IRC')
df_wr = nonprofit_groups.get_group('WR')
df_dr = nonprofit_groups.get_group('DR')
df_ci = nonprofit_groups.get_group('CI')
#next step is to understand each nonprofit data frame and pottential unqiue leads and traits


##################################################
#IRC - International Rescue Commmittee
df_irc.shape
df_irc.describe()
#heatmap
sns.set(style='whitegrid')
plt.figure(figsize=(14,10))
#dropping ID, nonprofit, day and date; will do boolean days 
#drop bible_quote because IRC was not a christian org so no observations
#IRC only had one major_news post so no correlations could be made with it; dropped
df_irc_without_variables = df_irc.drop(['nonprofit', 'ID', 'date', 'day', 'month', 'bible_quote', 'major_news'], axis=1)
sns.heatmap(df_irc_without_variables.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('Correlation Matrix Heatmap (IRC)', fontsize=18)
plt.savefig('Correlation Matrix Heatmap (IRC)')
plt.show
#interesting leads
# 1) likes correlation to indv_quote (.60), graphic (.20), partnership (.16)
# 2) comments corrleation to holiday (.42), hashtags (.29)
# 3) graphic_image correlation to persons_need (.58) and org persons (.47)
# 4) per_story correlation to persons_need (.50)

#find correlations with days of week
#using pd.get_dummies to turn Days into boolean to seperate IRC dataframe
df_irc_dummies = pd.get_dummies(df_irc, columns=['day'])
#make sure changes took place
df_irc_dummies.dtypes
#no need to drop days because dummies does already
df_irc_dummies_without_variables= df_irc_dummies.drop(['nonprofit', 'ID', 'date', 'month'], axis=1)
#print correlation matrix
irc_corr_matrix = df_irc_dummies_without_variables.corr()
irc_like_corr = irc_corr_matrix['likes']
irc_comment_corr = irc_corr_matrix['comments']
print(irc_like_corr)
print(irc_comment_corr)
#findings:
# 1) tue/likes .31 2) mon/comments .36

#use pearson test to determine statistical significance
# 1) tue/likes .31
irc_tue = df_irc_dummies['day_Tue']
irc_likes = df_irc_dummies['likes']
corr_coef, p_value =pearsonr(irc_tue, irc_likes)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
#Statistically signiciant
# 2) mon/comments .36
irc_mon = df_irc_dummies['day_Mon']
irc_comments = df_irc_dummies['comments']
corr_coef, p_value =pearsonr(irc_mon, irc_comments)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
#Statistically significant
#moderate positive relationship both with tue+likes and mon+comments

#explore correlations to engagements/pearsons
#likes+indv_quote, graphic, partnership
irc_likes = df_irc_dummies['likes']
irc_indv_quote = df_irc_dummies['indv_quote']
irc_graphic = df_irc_dummies['graphic']
irc_partnership = df_irc_dummies['partnership']
corr_likes_indv_quote, p_value_likes_indv_quote = pearsonr(irc_likes, irc_indv_quote)
corr_likes_graphic, p_value_likes_graphic = pearsonr(irc_likes, irc_graphic)
corr_likes_partnership, p_value_likes_partnership = pearsonr(irc_likes, irc_partnership)
print("Pearson's Correlation between 'likes' and 'indv_quote':", corr_likes_indv_quote)
print("P-value for 'likes' and 'indv_quote':", p_value_likes_indv_quote)
print("Pearson's Correlation between 'likes' and 'graphic':", corr_likes_graphic)
print("P-value for 'likes' and 'graphic':", p_value_likes_graphic)
print("Pearson's Correlation between 'likes' and 'partnership':", corr_likes_partnership)
print("P-value for 'likes' and 'partnership':", p_value_likes_partnership)
#strong positive relationship between likes and indv_quote

#comments+holiday, hashtags
irc_comments = df_irc_dummies['comments']
irc_holiday = df_irc_dummies['holiday']
irc_hashtags = df_irc_dummies['hashtags']
corr_comments_holiday, p_value_comments_holiday = pearsonr(irc_comments, irc_holiday)
corr_comments_hashtags, p_value_comments_hashtags = pearsonr(irc_comments, irc_hashtags)
print("Pearson's Correlation between 'comments' and 'holiday':", corr_comments_holiday)
print("P-value for 'comments' and 'holiday':", p_value_comments_holiday)
print("Pearson's Correlation between 'comments' and 'hashtags':", corr_comments_hashtags)
print("P-value for 'comments' and 'hashtags':", p_value_comments_hashtags)
#moderate positive relationship between comments and holiday

#explore percentage break down of post_type 1s/2s on graphics
#number of graphic photos
df_irc_graphics_1 = df_irc[df_irc['graphic']==1]
print("Number of Graphics on IRCs Instagram: " + str(df_irc_graphics_1.shape[0]))
df_irc_graphics_post_type_1 = df_irc_graphics_1[df_irc_graphics_1['post_type']==1]
print("Number of Photos with Graphics on IRCs Instagram: " + str(df_irc_graphics_post_type_1.shape[0]))
#now ratio
df_irc_ratio_graphic_post_type_1 = (df_irc_graphics_post_type_1.shape[0] / df_irc_graphics_1.shape[0])
print("Percent of photos that are graphics: " +str(df_irc_ratio_graphic_post_type_1))
#percent of overall posts that are graphics
df_irc_overall_posts_number = df_irc.shape[0]
irc_ratio_graphic_to_overall = ((df_irc_graphics_1.shape[0]) / df_irc_overall_posts_number)
print('Percent of posts that were graphics: ' + str(irc_ratio_graphic_to_overall))
#number of graphic carousels
df_irc_graphic_post_type_2 = df_irc_graphics_1[df_irc_graphics_1['post_type']==2]
print("Number of Carousels with Graphics on IRCs Instagram: " + str(df_irc_graphic_post_type_2.shape[0]))
#now ratio
df_irc_ratio_graphic_post_type_2 = (df_irc_graphic_post_type_2.shape[0] / df_irc_graphics_1.shape[0])
print("Percent of carousels that are graphics: " + str(df_irc_ratio_graphic_post_type_2 ))

#make donut for graphic posts split up for report
#data for the donut plot (overall posts vs graphics
df_irc_graphics_0_3 = df_irc[(df_irc['graphic'] == 0) | (df_irc['graphic'] == 3)]
num_posts_not_graphics = df_irc_graphics_0_3.shape[0]
num_graphics = df_irc_graphics_1.shape[0]
ci_total_posts = num_posts_not_graphics + num_graphics
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFC0CB','#AFEEEE']
wedges, _ = ax.pie([num_graphics, num_posts_not_graphics], labels=['Graphics', 'Not/Other Post'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Share of Graphics among Posts", fontsize=18)
plt.savefig('Donut Graphics vs overall posts (IRC)')
plt.show()

#make donut for graphic posts split up for report
# Data for the donut plot
num_photos = df_irc_graphics_post_type_1.shape[0]
num_carousels = df_irc_graphic_post_type_2.shape[0]
total_graphics = num_photos + num_carousels
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#90EE90','#F08080']
wedges, _ = ax.pie([num_photos, num_carousels], labels=['Photos', 'Carousels'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Photos vs. Carousels", fontsize=18)
plt.savefig('Donut Graphics carousels vs photos (IRC)')
plt.show()

#explore percentage break down of persons_needs and org_persons in graphic images
#number of graphic with images
df_irc_graphic_image_1 = df_irc[df_irc['graphic_with_image']==1]
print("Number of times there is a Graphic with an Image on IRCs Instagram: " + str(df_irc_graphic_image_1.shape[0]))
#number of persons_need in graphic images
irc_persons_need_1 = df_irc_graphic_image_1[df_irc_graphic_image_1['persons_in_need']==1].shape[0]
print("Number of times there is a graphic with an image of persons in need: " + str(irc_persons_need_1))
#ratio 
irc_ratio_persons_need_in_graphic = (irc_persons_need_1 / df_irc_graphic_image_1.shape[0])
print("Ratio of Persons in Need in Graphics: " + str(irc_ratio_persons_need_in_graphic))
#23% of graphics with images have persons in needs
#number of org_persons in graphic images
irc_org_persons_1 = df_irc_graphic_image_1[df_irc_graphic_image_1['org_persons']==1].shape[0]
print("Number of times there is a graphic with Org Persons: " + str(irc_org_persons_1))
#None of the graphics had people from IRC in them
#percentage of graphics that had images in them
irc_graphic_images_to_graphic = (df_irc_graphic_image_1.shape[0] / df_irc_graphics_1.shape[0])
print("Percentage of the Graphics had images: " + str(irc_graphic_images_to_graphic))

#make donut for graphic with image breakdown report
#Data for the donut plot
irc_number_graphic_image = df_irc_graphic_image_1.shape[0]
df_irc_number_not_image = df_irc_graphics_1[df_irc_graphics_1['graphic_with_image']==0]
irc_number_graphic_not_image = df_irc_number_not_image.shape[0]
total_graphics = irc_number_graphic_not_image + irc_number_graphic_image
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#B03060','#40E0D0']
wedges, _ = ax.pie([irc_number_graphic_not_image, irc_number_graphic_image], labels=['No Image', 'Image'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Real Photo Image Included", fontsize=18)
plt.savefig('Donut Graphics image vs not (IRC)')
plt.show()


#explore percentage breakdown of of persons_needs and org_persons in per_story
#number of personal stories
df_irc_per_story_1 = df_irc[df_irc['per_story']==1]
print("Number of Personal Story Posts on IRCs Instagram: " + str(df_irc_per_story_1.shape[0]))
#number of persons_needs personal stories
df_irc_persons_needs_per_story = df_irc_per_story_1[df_irc_per_story_1['persons_in_need']==1]
print("Number of Person in Need's Personal Story Posts: " + str(df_irc_persons_needs_per_story.shape[0]))
#Ratio
df_irc_ratio_persons_needs_story = (df_irc_persons_needs_per_story.shape[0] / df_irc_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_irc_ratio_persons_needs_story)) 
#number of org_persons personal stories
df_irc_org_persons_per_story = df_irc_per_story_1[df_irc_per_story_1['org_persons']==1]
print("Number of Org Persons's Personal Story Posts: " + str(df_irc_org_persons_per_story.shape[0]))
#Ratio
df_irc_ratio_org_persons_story = (df_irc_org_persons_per_story.shape[0] / df_irc_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_irc_ratio_org_persons_story))

#make donut for personal story posts vs overall posts
#Data for the donut plot
df_irc_post_not_per = df_irc[df_irc['per_story']==0]
irc_overall_posts_not_personal_story = df_irc_post_not_per.shape[0]
irc_number_per_story = df_irc_per_story_1.shape[0]
total_graphics = irc_overall_posts_not_personal_story + irc_number_per_story
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFA500','#0000FF']
wedges, _ = ax.pie([irc_overall_posts_not_personal_story, irc_number_per_story], labels=['Not/Other Post', 'Personal Story'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Posts: Personal Story Posts", fontsize=18)
plt.savefig('Donut Presonal Story vs overall (IRC)')
plt.show()

#find median enagement of personal story posts vs not
irc_median_likes_not_story = df_irc_post_not_per['likes'].median()
irc_median_likes_story = df_irc_per_story_1['likes'].median()
print('Not personal story median likes: ' + str(irc_median_likes_not_story))
print('Personal Story median likes: ' + str(irc_median_likes_story))
#comments
irc_median_comments_not_story= df_irc_post_not_per['comments'].median()
irc_median_comments_story = df_irc_per_story_1['comments'].median()
print('Not personal story median comments: ' + str(irc_median_comments_not_story))
print('Personal Story median comments: ' + str(irc_median_comments_story ))

#distribution of posts by days for irc
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.countplot(x='day', data=df_irc, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Post by Day Posted (IRC)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Distribution of Posts Analyzed by Day Posted (IRC)')
plt.show()
#most frequent day- Friday

#donut of 3 different post types
df_irc_post_type_count = df_irc['post_type'].value_counts()
df_irc_post_type_count
labels = ['Carousel', 'Video', 'Photo']
irc_sizes = df_irc_post_type_count.values
colors = {'Carousel': 'skyblue', 'Video': 'lightcoral', 'Photo': 'lightgreen'}
plt.figure(figsize=(10, 6))
wedges, _, autotexts = plt.pie(irc_sizes, labels=labels, colors=[colors[label] for label in labels], textprops={'fontsize':16},
                               autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(irc_sizes))})', startangle=90, wedgeprops=dict(width=0.8, edgecolor='w'))
#appearance of the text
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_color('black')
#white circle at the center to make it a donut plot
inner_circle = plt.Circle((0, 0), 0.4, color='white')
plt.gca().add_artist(inner_circle)
plt.title('Distribution of Post Types (IRC)', fontsize=20)
plt.axis('equal')
plt.savefig('Donut Post Types for IRC')
plt.show()

#now looking at comparison of likes/comments across days and post_types
#a lot of consideration whether to use mean, median, or weighted
#some huge numbers skew the data heavily that the z score did not catch
#while not outliers they do skew the averages heavily
#decided medians is best course to attempt to show how the typical post will perform, not viral ones

#Creating median likes for post_types plot
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue','Video': 'lightcoral'}
irc_median_likes_per_post_type = df_irc.groupby('post_type')['likes'].median()
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_irc['post_type'], y=df_irc['likes'], palette=colors_dict.values())
#print median value on each box
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Likes Count', fontsize=16)
plt.title('Median Likes for Each Post Type (IRC)', fontsize=20)
plt.ylim(0,3000)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Likes for Post Type (IRC)')
plt.show()

#Creating median comments for post_types plot
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue','Video': 'lightcoral'}
irc_median_comments_per_post_type = df_irc.groupby('post_type')['comments'].median()
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x=df_irc['post_type'], y=df_irc['comments'], palette=colors_dict.values())
#print median value on each box
medians = df_irc.groupby('day')['comments'].median()
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Comments Count', fontsize=16)
plt.title('Median Comments for Each Post Type (IRC)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,130)
plt.savefig('Median Comments for Post Type (IRC)')
plt.show()

#calculate and graph median likes per day to find best days, will use later to compare to other nonprofits
irc_median_likes_per_day = df_irc.groupby('day')['likes'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_irc['day'], y=df_irc['likes'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_irc.groupby('day')['likes'].median()
for idx, day in enumerate(irc_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, irc_median_likes_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Median Likes Count', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylim(0,5500)
plt.title('Median Likes for Each Day (IRC)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Likes for Each Day(IRC)')
plt.show()

#do the same for median comments
irc_median_comments_per_day = df_irc.groupby('day')['comments'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_irc['day'], y=df_irc['comments'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_irc.groupby('day')['comments'].median()
for idx, day in enumerate(irc_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, irc_median_comments_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Median Comments Count', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylim(0,100)
plt.title('Median Comments for Each Day (IRC)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Comments for Each Day(IRC)')
plt.show()

#going to now visualize emoji and hashtag usage
#creating a violin for emojis and post type
irc_emojis_by_post= df_irc.groupby('post_type')['emojis'].mean()
plt.figure(figsize=(10,6))
colors = ["lightgreen", "skyblue", "lightcoral"]
sns.violinplot(x='post_type', y='emojis', data=df_irc, palette=colors)
plt.xlabel('Post Type', fontsize=16)
plt.ylabel('Average Emojis', fontsize=16)
#to avoid type error
post_type_labels = ['Photo', 'Carousel', 'Video']
plt.xticks(ticks=[0, 1, 2], labels=post_type_labels, fontsize=14)
plt.title('Usage of Emojis by Post Type (IRC)', fontsize=20)
plt.ylim((0,6))
plt.tight_layout()
plt.savefig('Violin Plot Emojis by Post Type (IRC)')
plt.show()

#creating violin plot for hashtags and post type
irc_hashtags_by_post= df_irc.groupby('post_type')['hashtags'].mean()
plt.figure(figsize=(10,6))
colors = ["lightgreen", "skyblue", "lightcoral"]
sns.violinplot(x='post_type', y='hashtags', data=df_irc, palette=colors)
plt.xlabel('Post Type', fontsize=16)
plt.ylabel('Average Hashtags', fontsize=16)
#to avoid type error
post_type_labels = ['Photo', 'Carousel', 'Video']
plt.xticks(ticks=[0, 1, 2], labels=post_type_labels, fontsize=14)
plt.title('Usage of Hashtags by Post Type (IRC)', fontsize=20)
plt.ylim((0,5))
plt.tight_layout()
plt.savefig('Violin Plot Hashtags by Post Type (IRC)')
plt.show()

#from heatmap looking at correlations with emoji/hashtags
#emojis+hashtags (.22), emojis+per_story(.21), emojis+graphic_image(.19)
#hashtags+graphic_image(.21), hashtags+per_story

#pearsons test for correlations with Emojis/Hashtags
#emojis+hashtags
irc_emojis = df_irc['emojis']
irc_hashtags = df_irc['hashtags']
correlation_coefficient, p_value = pearsonr(irc_emojis, irc_hashtags)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#emojis+per_story
irc_per_story =  df_irc['per_story']
correlation_coefficient, p_value = pearsonr(irc_emojis, irc_per_story)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#emojis+graphic_image
irc_graphic_image = df_irc['graphic_with_image']
correlation_coefficient, p_value = pearsonr(irc_emojis, irc_graphic_image)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtags+graphic_image
correlation_coefficient, p_value = pearsonr(irc_hashtags, irc_graphic_image)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtags+per_story
correlation_coefficient, p_value = pearsonr(irc_hashtags, irc_per_story)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#overall no statistical significance with any besides hashtags+graphics, which is weak/negligible

#looking at percentage of graphics that have a statistic present
df_irc_statistic_1 = df_irc[df_irc['statistic']==1]
df_irc_statistic_1
#1 statistic post
df_irc_graphics_1 = df_irc[df_irc['graphic']==1]
irc_stat_in_graphic = df_irc_graphics_1[df_irc_graphics_1['statistic']==1]
irc_stat_in_graphic
#the one stat post is in a graphic

#look at percentage of videos that were short form
df_irc_videos= df_irc[df_irc['post_type']==3]
df_irc_short_form_vid = df_irc_videos[df_irc_videos['short_form_vid']==1]
df_irc_long_form_vid = df_irc_videos[df_irc_videos['short_form_vid']==0]
irc_number_short_vid= len(df_irc_short_form_vid)
irc_number_videos = len(df_irc_videos)
#calculate ratio
irc_ratio_short_vids = (irc_number_short_vid / irc_number_videos)
print('Ratio of Videos that are short form: ' + str(irc_ratio_short_vids))

#make donut visual for report
#break down short videos vs regular videos
irc_number_short_vid = df_irc_short_form_vid.shape[0]
irc_number_long_vid = df_irc_long_form_vid.shape[0]
irc_total_videos = irc_number_short_vid + irc_number_long_vid
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#4682B4', '#E3D6BF']
wedges, _ = ax.pie([irc_number_short_vid, irc_number_long_vid], labels=['Tik Tok Style', 'Regular Style'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Video Posts: Tik Tok Style vs. Regular Style", fontsize=18)
plt.savefig('Donut Videos tik toks vs regular (IRC)')
plt.show()

#calculate median engagement of videos and short videos
#likes
irc_median_likes_short_vid = df_irc_short_form_vid['likes'].median()
#have to create data frame of just long videos
df_irc_regular_vid = df_irc_videos[df_irc_videos['short_form_vid']==0]
irc_median_likes_regular_vid = df_irc_regular_vid['likes'].median()
print('Short form vid median likes: ' + str(irc_median_likes_short_vid))
print('Regular form vid median likes: ' + str(irc_median_likes_regular_vid))
#comments
irc_median_comments_short_vid = df_irc_short_form_vid['comments'].median()
irc_median_comments_regular_vid = df_irc_regular_vid['comments'].median()
print('Short form vid median comments: ' + str(irc_median_comments_short_vid))
print('Regular form vid median comments: ' + str(irc_median_comments_regular_vid))

#how many posts had indiviudal quotes
df_irc_indv_quote = df_irc[df_irc['indv_quote']!=0]
print("Number of posts with Indivual Quote featured: " + str(df_irc_indv_quote.shape[0]))
#3 posts with individual quotes

#median likes overall
irc_median_likes = df_irc['likes'].median()
print(irc_median_likes)
irc_median_comments = df_irc['comments'].median()
print(irc_median_comments)

#median likes/comments for only graphics posts
irc_median_likes_graphics = df_irc_graphics_1['likes'].median()
print(irc_median_likes_graphics)
irc_median_comments_graphics = df_irc_graphics_1['comments'].median()
print(irc_median_comments_graphics)


##################################################
#CI - Compassion International
df_ci.shape
df_ci.count()

#check how many major news posts there are to see if able to include in heatmap
df_ci_major_news_count= df_ci[df_ci['major_news']==1]
df_ci_major_news_count
#no major news posts so excluded from heat map

#heatmap
sns.set(style='whitegrid')
plt.figure(figsize=(14,10))
#dropping ID, nonprofit, day and date; will do boolean days
#include bible_quote this time because CI is a religious org
df_ci_without_variables = df_ci.drop(['nonprofit', 'ID', 'date', 'day', 'month', 'major_news'], axis=1)
sns.heatmap(df_ci_without_variables.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('Correlation Matrix Heatmap (CI)', fontsize=18)
plt.savefig('Correlation Matrix Heatmap (CI)')
plt.show
#interesting leads:
#likes+per_story(.18), likes+holiday(.20), likes+partnership(.19), likes+comments(.54), comments+hashtags(.15)

#create boolean days to findd potential correlations
df_ci_dummies = pd.get_dummies(df_ci, columns=['day'])
df_ci_dummies.dtypes
df_ci_dummies_without_variables= df_ci_dummies.drop(['nonprofit', 'ID', 'date', 'month'], axis=1)
ci_corr_matrix = df_ci_dummies_without_variables.corr()
ci_like_corr = ci_corr_matrix['likes']
ci_comment_corr = ci_corr_matrix['comments']
print(ci_like_corr)
print(ci_comment_corr)
#potential correlation with mon+likes and sat+comments

#pearsons test to determine validity
#use pearson test to determine statistical significance
# 1) tue/likes .13
ci_tue = df_ci_dummies['day_Tue']
ci_likes = df_ci_dummies['likes']
corr_coef, p_value =pearsonr(ci_tue, ci_likes)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
#weak
# 2) sat/comments .28
ci_sat = df_ci_dummies['day_Sat']
ci_comments = df_ci_dummies['comments']
corr_coef, p_value =pearsonr(ci_sat, ci_comments)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
#correlation between sat and comments is statistically significant

#explore correlation leads to engagements/pearsons
#likes+per_story, holiday, partnership, comments
#comments+hashtags
ci_likes = df_ci_dummies['likes']
ci_per_story = df_ci_dummies['per_story']
ci_holiday = df_ci_dummies['holiday']
ci_partnership = df_ci_dummies['partnership']
ci_comments = df_ci_dummies['comments']
ci_hashtags = df_ci_dummies['hashtags']
corr_likes_per_story, p_value_likes_per_story = pearsonr(ci_likes, ci_per_story)
corr_likes_holiday, p_value_likes_holiday = pearsonr(ci_likes, ci_holiday)
corr_likes_partnership, p_value_likes_partnership = pearsonr(ci_likes, ci_partnership)
corr_likes_comments, p_value_likes_comments = pearsonr(ci_likes, ci_comments)
corr_comments_hashtags, p_value_comments_hashtags = pearsonr(ci_likes, ci_hashtags)
print("Correlation and P-values:")
print(f"Likes vs. Per_Story: Correlation = {corr_likes_per_story}, P-value = {p_value_likes_per_story}")
print(f"Likes vs. Holiday: Correlation = {corr_likes_holiday}, P-value = {p_value_likes_holiday}")
print(f"Likes vs. Partnership: Correlation = {corr_likes_partnership}, P-value = {p_value_likes_partnership}")
print(f"Likes vs. Comments: Correlation = {corr_likes_comments}, P-value = {p_value_likes_comments}")
print(f"Comments vs. Hashtags: Correlation = {corr_comments_hashtags}, P-value = {p_value_comments_hashtags}")
#only strong takeaway is that likes and comments are statistically significant

#explore percentage break down of post_type 1s/2s on CI graphics
#number of graphic photos
df_ci_graphics_1 = df_ci[df_ci['graphic']==1]
print("Number of Graphics on CIs Instagram: " + str(df_ci_graphics_1.shape[0]))
df_ci_graphics_post_type_1 = df_ci_graphics_1[df_ci_graphics_1['post_type']==1]
print("Number of Photos with Graphics on CIs Instagram: " + str(df_ci_graphics_post_type_1.shape[0]))
#now ratio
df_ci_ratio_graphic_post_type_1 = (df_ci_graphics_post_type_1.shape[0] / df_ci_graphics_1.shape[0])
print("Percent of photos that are graphics: " +str(df_ci_ratio_graphic_post_type_1))
#percent of overall posts that are graphics
df_ci_overall_posts_number = df_ci.shape[0]
ci_ratio_graphic_to_overall = ((df_ci_graphics_1.shape[0]) / df_ci_overall_posts_number)
print('Percent of posts that were graphics: ' + str(ci_ratio_graphic_to_overall))
#number of graphic carousels
df_ci_graphic_post_type_2 = df_ci_graphics_1[df_ci_graphics_1['post_type']==2]
print("Number of Carousels with Graphics on CIs Instagram: " + str(df_ci_graphic_post_type_2.shape[0]))
#now ratio
df_ci_ratio_graphic_post_type_2 = (df_ci_graphic_post_type_2.shape[0] / df_ci_graphics_1.shape[0])
print("Percent of carousels that are graphics: " + str(df_ci_ratio_graphic_post_type_2 ))

#calculate median engagement of graphic photos and carousels
#likes
ci_median_likes_graphic_photo = df_ci_graphics_post_type_1['likes'].median()
ci_median_likes_graphic_carousel = df_ci_graphic_post_type_2['likes'].median()
print('Graphic photo median likes: ' + str(ci_median_likes_graphic_photo))
print('Graphic carousel median likes: ' + str(ci_median_likes_graphic_carousel))
#comments
ci_median_comments_graphic_photo = df_ci_graphics_post_type_1['comments'].median()
ci_median_comments_graphic_carousel = df_ci_graphic_post_type_2['comments'].median()
print('Graphic photo median comments: ' + str(ci_median_comments_graphic_photo))
print('Graphic carousel median comments: ' + str(ci_median_comments_graphic_carousel))

#make donut for graphic posts split up for report
#data for the donut plot (overall posts vs graphics)
df_ci_graphics_0_3 = df_ci[(df_ci['graphic'] == 0) | (df_ci['graphic'] == 3)]
num_posts_not_graphics = df_ci_graphics_0_3.shape[0]
num_graphics = df_ci_graphics_1.shape[0]
ci_total_posts = num_posts_not_graphics + num_graphics
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFC0CB','#AFEEEE']
wedges, _ = ax.pie([num_graphics, num_posts_not_graphics], labels=['Graphics', 'Not/Other Post'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Share of Graphics among Posts", fontsize=18)
plt.savefig('Donut Graphics vs overall posts (CI)')
plt.show()

#make donut for graphic posts split up for report
#data for the donut plot (graphics carousels vs photos)
num_photos = df_ci_graphics_post_type_1.shape[0]
num_carousels = df_ci_graphic_post_type_2.shape[0]
total_graphics = num_photos + num_carousels
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#90EE90','#F08080']
wedges, _ = ax.pie([num_photos, num_carousels], labels=['Photos', 'Carousels'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Photos vs. Carousels", fontsize=18)
plt.savefig('Donut Graphics carousels vs photos (CI)')
plt.show()

#explore percentage break down of persons_needs and org_persons in graphic images
#number of graphic with images
df_ci_graphic_image_1 = df_ci[df_ci['graphic_with_image']==1]
print("Number of times there is a Graphic with an Image on CIs Instagram: " + str(df_ci_graphic_image_1.shape[0]))
#number of persons_need in graphic images
ci_persons_need_1 = df_ci_graphic_image_1[df_ci_graphic_image_1['persons_in_need']==1].shape[0]
print("Number of times there is a graphic with an image of persons in need: " + str(ci_persons_need_1))
#ratio 
ci_ratio_persons_need_in_graphic = (ci_persons_need_1 / df_ci_graphic_image_1.shape[0])
print("Ratio of Persons in Need in Graphics: " + str(ci_ratio_persons_need_in_graphic))
#number of org_persons in graphic images
ci_org_persons_1 = df_ci_graphic_image_1[df_ci_graphic_image_1['org_persons']==1].shape[0]
print("Number of times there is a graphic with Org Persons: " + str(ci_org_persons_1))
#percentage of graphics that had images in them
ci_graphic_images_to_graphic = (df_ci_graphic_image_1.shape[0] / df_ci_graphics_1.shape[0])
print("Percentage of the Graphics had images: " + str(ci_graphic_images_to_graphic))

#make donut for graphic with image breakdown report
#Data for the donut plot (graphics vs graphics with images)
ci_number_graphic_image = df_ci_graphic_image_1.shape[0]
df_ci_number_not_image = df_ci_graphics_1[df_ci_graphics_1['graphic_with_image']==0]
ci_number_graphic_not_image = df_ci_number_not_image.shape[0]
total_graphics = ci_number_graphic_not_image + ci_number_graphic_image
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#B03060','#40E0D0']
wedges, _ = ax.pie([ci_number_graphic_not_image, ci_number_graphic_image], labels=['No Image', 'Image'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Real Photo Image Included", fontsize=18)
plt.savefig('Donut Graphics image vs not (CI)')
plt.show()


#explore percentage breakdown of of persons_needs and org_persons in per_story for CI
#number of personal stories
df_ci_per_story_1 = df_ci[df_ci['per_story']==1]
print("Number of Personal Story Posts on CIs Instagram: " + str(df_ci_per_story_1.shape[0]))
#number of persons_needs personal stories
df_ci_persons_needs_per_story = df_ci_per_story_1[df_ci_per_story_1['persons_in_need']==1]
print("Number of Person in Need's Personal Story Posts: " + str(df_ci_persons_needs_per_story.shape[0]))
#Ratio
df_ci_ratio_persons_needs_story = (df_ci_persons_needs_per_story.shape[0] / df_ci_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_ci_ratio_persons_needs_story)) 
#number of org_persons personal stories
df_ci_org_persons_per_story = df_ci_per_story_1[df_ci_per_story_1['org_persons']==1]
print("Number of Org Persons's Personal Story Posts: " + str(df_ci_org_persons_per_story.shape[0]))
#Ratio
df_ci_ratio_org_persons_story = (df_ci_org_persons_per_story.shape[0] / df_ci_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_ci_ratio_org_persons_story))
#Ratio of personal story to overall posts
df_ci_overall_posts_number = df_ci.shape[0]
ci_ratio_per_story_to_overall = ((df_ci_per_story_1.shape[0]) / df_ci_overall_posts_number)
print('Percentage of overall posts that are personal stories: ' + str(ci_ratio_per_story_to_overall))

#make donut for personal story posts vs overall posts
#Data for the donut plot
df_ci_post_not_per = df_ci[df_ci['per_story']==0]
ci_overall_posts_not_personal_story = df_ci_post_not_per.shape[0]
ci_number_per_story = df_ci_per_story_1.shape[0]
total_graphics = ci_overall_posts_not_personal_story + ci_number_per_story
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFA500','#0000FF']
wedges, _ = ax.pie([ci_overall_posts_not_personal_story, ci_number_per_story], labels=['Not/Other Post', 'Personal Story'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Posts: Personal Story Posts", fontsize=18)
plt.savefig('Donut Presonal Story vs overall (CI)')
plt.show()

#find median enagement of personal story posts vs not
ci_median_likes_not_story = df_ci_post_not_per['likes'].median()
ci_median_likes_story = df_ci_per_story_1['likes'].median()
print('Not personal story median likes: ' + str(ci_median_likes_not_story))
print('Personal Story median likes: ' + str(ci_median_likes_story))
#comments
ci_median_comments_not_story= df_ci_post_not_per['comments'].median()
ci_median_comments_story = df_ci_per_story_1['comments'].median()
print('Not personal story median comments: ' + str(ci_median_comments_not_story))
print('Personal Story median comments: ' + str(ci_median_comments_story ))


#distribution of CI posts per day
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.countplot(x='day', data=df_ci, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Post by Day Posted (CI)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Distribution of Posts Analyzed by Day Posted (CI)')
plt.show()

#donut of CI distribution of 3 different post types
df_ci_post_type_count = df_ci['post_type'].value_counts()
df_ci_post_type_count
labels = ['Video', 'Carousel', 'Photo']
colors = {'Carousel': 'skyblue', 'Video': 'lightcoral', 'Photo': 'lightgreen'}
ci_sizes = df_ci_post_type_count.values
plt.figure(figsize=(10, 6))
wedges, _, autotexts = plt.pie(ci_sizes, labels=labels, colors=[colors[label] for label in labels], textprops={'fontsize':16},
                               autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(ci_sizes))})', startangle=90, wedgeprops=dict(width=0.8, edgecolor='w'))
#appearance of the text
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_color('black')
#white circle at the center to make it a donut plot
inner_circle = plt.Circle((0, 0), 0.4, color='white')
plt.gca().add_artist(inner_circle)
plt.title('Distribution of Post Types (CI)', fontsize=20)
plt.axis('equal')
plt.savefig('Donut Post Types for CI')
plt.show()


#now looking at comparison of likes/comments across days and post_types
#Creating median likes for post_types plot
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue', 'Video': 'lightcoral'}
ci_median_likes_per_post_type = df_ci.groupby('post_type')['likes'].median()
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_ci['post_type'], y=df_ci['likes'], palette=colors_dict.values())
#print median value on each box
medians = df_ci.groupby('day')['likes'].median()
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Likes Count', fontsize=16)
plt.title('Median Likes for Each Post Type (CI)', fontsize=20)
plt.ylim(0,2000)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Likes for Post Type (CI)')
plt.show()

#Creating median comments for post_types plot
###
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue', 'Video': 'lightcoral'}
ci_median_comments_per_post_type = df_ci.groupby('post_type')['comments'].median()
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x=df_ci['post_type'], y=df_ci['comments'], palette=colors_dict.values())
#print median value on each box
medians = df_ci.groupby('day')['comments'].median()
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Comments Count', fontsize=16)
plt.title('Median Comments for Each Post Type (CI)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,50)
plt.savefig('Median Comments for Post Type (CI)')
plt.show()

#calculate/graph median likes to best days, will use later to compare to other nonprofits
ci_median_likes_per_day = df_ci.groupby('day')['likes'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_ci['day'], y=df_ci['likes'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_ci.groupby('day')['likes'].median()
for idx, day in enumerate(ci_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, ci_median_likes_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Median Likes Count', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.title('Median Likes for Each Day (CI)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,1600)
plt.savefig('Median Likes for Each Day(CI)')
plt.show()

#do the same for median comments
ci_median_comments_per_day = df_ci.groupby('day')['comments'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_ci['day'], y=df_ci['comments'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_ci.groupby('day')['comments'].median()
for idx, day in enumerate(ci_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, ci_median_comments_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Median Comments Count', fontsize=14)
plt.title('Median Comments for Each Day (CI)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,35)
plt.savefig('Median Comments for Each Day(CI)')
plt.show()

#going to now visualize emoji and hashtag usage
#creating a violin plot for emojis and post type
ci_emojis_by_post= df_ci.groupby('post_type')['emojis'].mean()
plt.figure(figsize=(10,6))
colors = ["lightgreen", "skyblue", "lightcoral"]
sns.violinplot(x='post_type', y='emojis', data=df_ci, palette=colors)
plt.xlabel('Post Type', fontsize=16)
plt.ylabel('Average Emojis', fontsize=16)
#to avoid type error
post_type_labels = ['Photo', 'Carousel', 'Video']
plt.xticks(ticks=[0, 1, 2], labels=post_type_labels, fontsize=14)
plt.title('Usage of Emojis by Post Type (CI)', fontsize=20)
plt.ylim((0,4))
plt.tight_layout()
plt.savefig('Violin Plot Emojis by Post Type (CI)')
plt.show()

#creating violin plot for hashtags and post type
ci_hashtags_by_post= df_ci.groupby('post_type')['hashtags'].mean()
plt.figure(figsize=(10,6))
colors = ["lightgreen", "skyblue", "lightcoral"]
sns.violinplot(x='post_type', y='hashtags', data=df_ci, palette=colors)
plt.xlabel('Post Type', fontsize=16)
plt.ylabel('Average Hashtags', fontsize=16)
#to avoid type error
post_type_labels = ['Photo', 'Carousel', 'Video']
plt.xticks(ticks=[0, 1, 2], labels=post_type_labels, fontsize=14)
plt.title('Usage of Hashtags by Post Type (CI)', fontsize=20)
plt.ylim((0,13))
plt.tight_layout()
plt.savefig('Violin Plot Hashtags by Post Type (CI)')
plt.show()

#look heatmap again to find pottential correlations with emojis/hashtags
#emojis+ graphic, hashtags, post type
#hashtags+ per_story, persons_in_need, stat, org_persons, graphic_with_image, post_type

#pearson to determine validity of correlations
#emoji+graphic
ci_emojis = df_ci['emojis']
ci_graphic = df_ci['graphic']
correlation_coefficient, p_value = pearsonr(ci_emojis, ci_graphic)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#emojis+hashtags
ci_hashtags =  df_ci['hashtags']
correlation_coefficient, p_value = pearsonr(ci_emojis, ci_hashtags)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#emojis+post_type
ci_post_type =  df_ci['post_type']
correlation_coefficient, p_value = pearsonr(ci_emojis, ci_post_type)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtag+per_story
ci_per_story =  df_ci['per_story']
correlation_coefficient, p_value = pearsonr(ci_hashtags, ci_per_story)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtags+persons_in_need
ci_persons_in_need =  df_ci['persons_in_need']
correlation_coefficient, p_value = pearsonr(ci_hashtags, ci_persons_in_need)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtags+statistic
ci_statistic =  df_ci['statistic']
correlation_coefficient, p_value = pearsonr(ci_hashtags, ci_statistic)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtag+org_persons
ci_org_persons =  df_ci['org_persons']
correlation_coefficient, p_value = pearsonr(ci_hashtags, ci_org_persons)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtag+graphic_with_image
ci_graphic_with_image =  df_ci['graphic_with_image']
correlation_coefficient, p_value = pearsonr(ci_hashtags, ci_graphic_with_image)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#hashtag+post_type
ci_post_type =  df_ci['post_type']
correlation_coefficient, p_value = pearsonr(ci_hashtags, ci_post_type)
print("Pearson correlation coefficient: " + str(correlation_coefficient))
print("P-value: " + str(p_value))
#significant takeaways: emoji+graphic, emojis+post_type, hashtags+org_persons, hashtag+post_type

#looking at percentage of graphics that have a statistic present
df_ci_statistic_1 = df_ci[df_ci['statistic']==1]
df_ci_statistic_1
#two statistic posts
df_ci_graphics_1 = df_ci[df_ci['graphic']==1]
ci_stat_in_graphic = df_ci_graphics_1[df_ci_graphics_1['statistic']==1]
ci_stat_in_graphic
#none were integrated with a graphic

#look at percentage of videos that were short form
df_ci_videos= df_ci[df_ci['post_type']==3]
df_ci_short_form_vid = df_ci_videos[df_ci_videos['short_form_vid']==1]
ci_number_short_vid= len(df_ci_short_form_vid)
ci_number_videos = len(df_ci_videos)
#calculate ratio
ci_ratio_short_vids = (ci_number_short_vid / ci_number_videos)
print('Ratio of Videos that are short form: ' + str(ci_ratio_short_vids))

#make donut visual for report
#break down short videos vs regular videos
df_ci_long_form_vid = df_ci_videos[df_ci_videos['short_form_vid']==0]
ci_number_short_vid = df_ci_short_form_vid.shape[0]
ci_number_long_vid = df_ci_long_form_vid.shape[0]
ci_total_videos = ci_number_short_vid + ci_number_long_vid
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#4682B4', '#E3D6BF']
wedges, _ = ax.pie([ci_number_short_vid, ci_number_long_vid], labels=['Tik Tok Style', 'Regular Style'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Video Posts: Tik Tok Style vs. Regular Style", fontsize=18)
plt.savefig('Donut Videos tik toks vs regular (CI)')
plt.show()

#calculate median engagement of videos and short videos
#likes
ci_median_likes_short_vid = df_ci_short_form_vid['likes'].median()
#have to create data frame of just long videos
df_ci_regular_vid = df_ci_videos[df_ci_videos['short_form_vid']==0]
ci_median_likes_regular_vid = df_ci_regular_vid['likes'].median()
print('Short form vid median likes: ' + str(ci_median_likes_short_vid))
print('Regular form vid median likes: ' + str(ci_median_likes_regular_vid))
#comments
ci_median_comments_short_vid = df_ci_short_form_vid['comments'].median()
ci_median_comments_regular_vid = df_ci_regular_vid['comments'].median()
print('Short form vid median comments: ' + str(ci_median_comments_short_vid))
print('Regular form vid median comments: ' + str(ci_median_comments_regular_vid))

#how many posts had indiviudal quotes
df_ci_indv_quote = df_ci[df_ci['indv_quote']!=0]
print("Number of posts with Indivual Quote featured: " + str(df_ci_indv_quote.shape[0]))
#1 post with individual quotes

#median likes and comments overall
ci_median_likes = df_ci['likes'].median()
print(ci_median_likes)
ci_median_comments = df_ci['comments'].median()
print(ci_median_comments)

#median likes/comments for only graphics posts
ci_median_likes_graphics = df_ci_graphics_1['likes'].median()
print(ci_median_likes_graphics)
ci_median_comments_graphics = df_ci_graphics_1['comments'].median()
print(ci_median_comments_graphics)


##################################################
#DR - Direct Relief
df_dr.shape
df_dr.count()
#smallest dataset of the 4

#check how many major news posts there are to see if able to include in heatmap
df_dr_major_news_count= df_ci[df_ci['major_news']==1]
df_dr_major_news_count
#no major news posts so excluded from heat map

#heatmap
sns.set(style='whitegrid')
plt.figure(figsize=(14,10))
#dropping ID, nonprofit, day and date; will do boolean days
#include bible_quote into voided variables because nonreligious org
df_dr_without_variables = df_dr.drop(['nonprofit', 'ID', 'date', 'day', 'month', 'bible_quote', 'major_news'], axis=1)
sns.heatmap(df_dr_without_variables.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('Correlation Matrix Heatmap (DR)', fontsize=18)
plt.savefig('Correlation Matrix Heatmap (DR)')
plt.show
#likes+graphic, comments, post_type & comments+partnership

#variables missing from matrix: holiday, call_action, emojis, indv_quote
#missing elements from the heatmap, check how many observation points are present for the variables missing
df_dr_emojis = df_dr[df_dr['emojis']!=0]
print('Amount of emoji posts used for DRs posts: ' + str(df_dr_emojis.shape[0]))
#no emojis used, explains the faulty plot from before
df_dr_holiday = df_dr[df_dr['holiday']!=0]
print('Amount of holiday posts used for DRs posts: ' + str(df_dr_holiday.shape[0]))
#0 holiday posts
df_dr_call_action = df_dr[df_dr['call_action']!=0]
print('Amount of call to action posts used for DRs posts: ' + str(df_dr_call_action.shape[0]))
#0 call to action posts
df_dr_indv_quote = df_dr[df_dr['indv_quote']!=0]
print('Amount of individual quote posts used for DRs posts: ' + str(df_dr_indv_quote.shape[0]))
#0 individual quote posts

#create boolean days to findd potential correlations
df_dr_dummies = pd.get_dummies(df_dr, columns=['day'])
df_dr_dummies.dtypes
df_dr_dummies_without_variables= df_dr_dummies.drop(['nonprofit', 'ID', 'date', 'month'], axis=1)
dr_corr_matrix = df_dr_dummies_without_variables.corr()
dr_like_corr = dr_corr_matrix['likes']
dr_comment_corr = dr_corr_matrix['comments']
print(dr_like_corr)
print(dr_comment_corr)
#potential correlation with thur+likes

#pearsons test to determine validity
#use pearson test to determine statistical significance
# 1) thur/likes .13
dr_tue = df_dr_dummies['day_Thur']
dr_likes = df_dr_dummies['likes']
corr_coef, p_value =pearsonr(dr_tue, dr_likes)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
#not statistically significant

#explore correlation leads to engagements/pearsons
#likes+graphic, comments, post_type
#comments+partnerships
dr_likes = df_dr_dummies['likes']
dr_graphic = df_dr_dummies['graphic']
dr_comments = df_dr_dummies['comments']
dr_post_type = df_dr_dummies['post_type']
dr_partnership = df_dr_dummies['partnership']
corr_likes_graphic, p_value_likes_graphic = pearsonr(dr_likes, dr_graphic)
corr_likes_comments, p_value_likes_comments = pearsonr(dr_likes, dr_comments)
corr_likes_post_type, p_value_likes_post_type = pearsonr(dr_likes, dr_post_type)
corr_comments_partnerships, p_value_comments_partnerships = pearsonr(dr_comments, dr_partnership)
print("Correlation and P-values:")
print(f"Likes vs. Graphic: Correlation = {corr_likes_graphic}, P-value = {p_value_likes_graphic}")
print(f"Likes vs. Comments: Correlation = {corr_likes_comments}, P-value = {p_value_likes_comments}")
print(f"Likes vs. Post Type: Correlation = {corr_likes_post_type}, P-value = {p_value_likes_post_type}")
print(f"Comments vs. Partnership: Correlation = {corr_comments_partnerships}, P-value = {p_value_comments_partnerships}")
#statsitically significant: likes+comments

#explore percentage break down of post_type 1s/2s on DR graphics
#number of graphic photos
df_dr_graphics_1 = df_dr[df_dr['graphic']==1]
print("Number of Graphics on DRs Instagram: " + str(df_dr_graphics_1.shape[0]))
df_dr_graphics_post_type_1 = df_dr_graphics_1[df_dr_graphics_1['post_type']==1]
print("Number of Photos with Graphics on DRs Instagram: " + str(df_dr_graphics_post_type_1.shape[0]))
#now ratio
df_dr_ratio_graphic_post_type_1 = (df_dr_graphics_post_type_1.shape[0] / df_dr_graphics_1.shape[0])
print("Percent of photos that are graphics: " +str(df_dr_ratio_graphic_post_type_1))
#percent of overall posts that are graphics
df_dr_overall_posts_number = df_dr.shape[0]
dr_ratio_graphic_to_overall = ((df_dr_graphics_1.shape[0]) / df_dr_overall_posts_number)
print('Percent of posts that were graphics: ' + str(dr_ratio_graphic_to_overall))
#number of graphic carousels
df_dr_graphic_post_type_2 = df_dr_graphics_1[df_dr_graphics_1['post_type']==2]
print("Number of Carousels with Graphics on CIs Instagram: " + str(df_dr_graphic_post_type_2.shape[0]))
#now ratio
df_dr_ratio_graphic_post_type_2 = (df_dr_graphic_post_type_2.shape[0] / df_dr_graphics_1.shape[0])
print("Percent of carousels that are graphics: " + str(df_dr_ratio_graphic_post_type_2 ))

#explore percentage break down of persons_needs and org_persons in graphic images
#number of graphic with images
df_dr_graphic_image_1 = df_dr[df_dr['graphic_with_image']==1]
print("Number of times there is a Graphic with an Image on DRs Instagram: " + str(df_dr_graphic_image_1.shape[0]))
#number of persons_need in graphic images
dr_persons_need_1 = df_dr_graphic_image_1[df_dr_graphic_image_1['persons_in_need']==1].shape[0]
print("Number of times there is a graphic with an image of persons in need: " + str(dr_persons_need_1))
#ratio 
dr_ratio_persons_need_in_graphic = (dr_persons_need_1 / df_dr_graphic_image_1.shape[0])
print("Ratio of Persons in Need in Graphics: " + str(dr_ratio_persons_need_in_graphic))
#number of org_persons in graphic images
dr_org_persons_1 = df_dr_graphic_image_1[df_dr_graphic_image_1['org_persons']==1].shape[0]
print("Number of times there is a graphic with Org Persons: " + str(dr_org_persons_1))
#percentage of graphics that had images in them
dr_graphic_images_to_graphic = (df_dr_graphic_image_1.shape[0] / df_dr_graphics_1.shape[0])
print("Percentage of the Graphics had images: " + str(dr_graphic_images_to_graphic))

#make donut for graphic with image breakdown report
#Data for the donut plot (graphics vs graphics with images)
dr_number_graphic_image = df_dr_graphic_image_1.shape[0]
df_dr_number_not_image = df_dr_graphics_1[df_dr_graphics_1['graphic_with_image']==0]
dr_number_graphic_not_image = df_dr_number_not_image.shape[0]
total_graphics = dr_number_graphic_not_image + dr_number_graphic_image
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#B03060','#40E0D0']
wedges, _ = ax.pie([dr_number_graphic_not_image, dr_number_graphic_image], labels=['No Image', 'Image'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Real Photo Image Included", fontsize=18)
plt.savefig('Donut Graphics image vs not (DR)')
plt.show()

#make donut for graphic posts split up for report
#data for the donut plot (overall posts vs graphics)
df_dr_graphics_0_3 = df_dr[(df_dr['graphic'] == 0) | (df_dr['graphic'] == 3)]
num_posts_not_graphics = df_dr_graphics_0_3.shape[0]
num_graphics = df_dr_graphics_1.shape[0]
dr_total_posts = num_posts_not_graphics + num_graphics
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFC0CB','#AFEEEE']
wedges, _ = ax.pie([num_graphics, num_posts_not_graphics], labels=['Graphics', 'Not/Other Post'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Share of Graphics among Posts", fontsize=18)
plt.savefig('Donut Graphics vs overall posts (DR)')
plt.show()

#make donut for graphic posts split up for report
#data for the donut plot (graphics carousels vs photos)
num_photos = df_dr_graphics_post_type_1.shape[0]
num_carousels = df_dr_graphic_post_type_2.shape[0]
total_graphics = num_photos + num_carousels
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#90EE90','#F08080']
wedges, _ = ax.pie([num_photos, num_carousels], labels=['Photos', 'Carousels'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Photos vs. Carousels", fontsize=18)
plt.savefig('Donut Graphics carousels vs photos (DR)')
plt.show()

#explore percentage breakdown of of persons_needs and org_persons in per_story for CI
#number of personal stories
df_dr_per_story_1 = df_dr[df_dr['per_story']==1]
print("Number of Personal Story Posts on DRs Instagram: " + str(df_dr_per_story_1.shape[0]))
#number of persons_needs personal stories
df_dr_persons_needs_per_story = df_dr_per_story_1[df_dr_per_story_1['persons_in_need']==1]
print("Number of Person in Need's Personal Story Posts: " + str(df_dr_persons_needs_per_story.shape[0]))
#Ratio
df_dr_ratio_persons_needs_story = (df_dr_persons_needs_per_story.shape[0] / df_dr_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_dr_ratio_persons_needs_story)) 
#number of org_persons personal stories
df_dr_org_persons_per_story = df_dr_per_story_1[df_dr_per_story_1['org_persons']==1]
print("Number of Org Persons's Personal Story Posts: " + str(df_dr_org_persons_per_story.shape[0]))
#Ratio
df_dr_ratio_org_persons_story = (df_dr_org_persons_per_story.shape[0] / df_dr_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_dr_ratio_org_persons_story))
#Ratio of personal story to overall posts
df_dr_overall_posts_number = df_dr.shape[0]
dr_ratio_per_story_to_overall = ((df_dr_per_story_1.shape[0]) / df_dr_overall_posts_number)
print('Percentage of overall posts that are personal stories: ' + str(dr_ratio_per_story_to_overall))

#make donut for personal story posts vs overall posts
#Data for the donut plot
df_dr_post_not_per = df_dr[df_dr['per_story']==0]
dr_overall_posts_not_personal_story = df_dr_post_not_per.shape[0]
dr_number_per_story = df_dr_per_story_1.shape[0]
total_graphics = dr_overall_posts_not_personal_story + dr_number_per_story
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFA500','#0000FF']
wedges, _ = ax.pie([dr_overall_posts_not_personal_story, dr_number_per_story], labels=['Not/Other Post', 'Personal Story'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Posts: Personal Story Posts", fontsize=18)
plt.savefig('Donut Presonal Story vs overall (DR)')
plt.show()

#find median enagement of personal story posts vs not
dr_median_likes_not_story = df_dr_post_not_per['likes'].median()
dr_median_likes_story = df_dr_per_story_1['likes'].median()
print('Not personal story median likes: ' + str(dr_median_likes_not_story))
print('Personal Story median likes: ' + str(dr_median_likes_story))
#comments
dr_median_comments_not_story= df_dr_post_not_per['comments'].median()
dr_median_comments_story = df_dr_per_story_1['comments'].median()
print('Not personal story median comments: ' + str(dr_median_comments_not_story))
print('Personal Story median comments: ' + str(dr_median_comments_story ))


#distribution of DR posts per day
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.countplot(x='day', data=df_dr, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Post by Day Posted (DR)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Distribution of Posts Analyzed by Day Posted (DR)')
plt.show()

#donut of DR distribution of 3 different post types
df_dr_post_type_count = df_dr['post_type'].value_counts()
df_dr_post_type_count
labels = ['Video', 'Carousel', 'Photo']
colors = {'Carousel': 'skyblue', 'Video': 'lightcoral', 'Photo': 'lightgreen'}
dr_sizes = df_dr_post_type_count.values
plt.figure(figsize=(10, 6))
wedges, _, autotexts = plt.pie(dr_sizes, labels=labels, colors=[colors[label] for label in labels], textprops={'fontsize':16},
                               autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(dr_sizes))})', startangle=90, wedgeprops=dict(width=0.8, edgecolor='w'))
#appearance of the text
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_color('black')
#white circle at the center to make it a donut plot
inner_circle = plt.Circle((0, 0), 0.4, color='white')
plt.gca().add_artist(inner_circle)
plt.title('Distribution of Post Types (DR)', fontsize=20)
plt.axis('equal')
plt.savefig('Donut Post Types for DR')
plt.show()

#now looking at comparison of likes/comments across days and post_types
#Creating median likes for post_types plot
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue', 'Video': 'lightcoral'}
dr_median_likes_per_post_type = df_dr.groupby('post_type')['likes'].median()
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_dr['post_type'], y=df_dr['likes'], palette=colors_dict.values())
#print median value on each box
medians = df_dr.groupby('day')['likes'].median()
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Likes Count', fontsize=16)
plt.title('Median Likes for Each Post Type (DR)', fontsize=20)
plt.ylim(0,150)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Likes for Post Type (DR)')
plt.show()

#Creating median comments for post_types plot
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue', 'Video': 'lightcoral'}
dr_median_comments_per_post_type = df_dr.groupby('post_type')['comments'].median()
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_dr['post_type'], y=df_dr['comments'], palette=colors_dict.values())
#print median value on each box
medians = df_dr.groupby('day')['comments'].median()
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Comments Count', fontsize=16)
plt.title('Median Comments for Each Post Type (DR)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Comments for Post Type (DR)')
plt.ylim(0,8)
plt.show()

#calculate/graph median likes to best days, will use later to compare to other nonprofits
dr_median_likes_per_day = df_dr.groupby('day')['likes'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_dr['day'], y=df_dr['likes'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_dr.groupby('day')['likes'].median()
for idx, day in enumerate(dr_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, dr_median_likes_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Median Likes Count', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.title('Median Likes for Each Day (DR)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,200)
plt.savefig('Median Likes for Each Day(DR)')
plt.show()

#do the same for median comments
dr_median_comments_per_day = df_dr.groupby('day')['comments'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_dr['day'], y=df_dr['comments'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_dr.groupby('day')['comments'].median()
for idx, day in enumerate(dr_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, dr_median_comments_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Median Comments Count', fontsize=14)
plt.title('Median Comments for Each Day (DR)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,6)
plt.savefig('Median Comments for Each Day(DR)')
plt.show()


#going to now visualize emoji and hashtag usage
#creating a violin plot for emojis and post type
dr_emojis_by_post= df_dr.groupby('post_type')['emojis'].mean()
plt.figure(figsize=(10,6))
colors = ["lightgreen", "skyblue", "lightcoral"]
sns.violinplot(x='post_type', y='emojis', data=df_dr, palette=colors)
plt.xlabel('Post Type', fontsize=16)
plt.ylabel('Average Emojis', fontsize=16)
#to avoid type error
post_type_labels = ['Photo', 'Carousel', 'Video']
plt.xticks(ticks=[0, 1, 2], labels=post_type_labels, fontsize=14)
plt.title('Usage of Emojis by Post Type (DR)', fontsize=20)
plt.tight_layout()
plt.show()

#code from before is not working, no observation points with emojis; hence faulty plot
#no emojis used, explains the faulty plot from before
#check for hashtags for dr
df_dr_hashtags = df_dr[df_dr['hashtags']!=0]
print('Amount of hashtags used for DRs posts: ' + str(df_dr_hashtags.shape[0]))
#only two hashtags used in all of their posts analyzed; not enough substantial data

#looking at percentage of graphics that have a statistic present
df_dr_statistic_1 = df_dr[df_dr['statistic']==1]
df_dr_statistic_1
#no statistic posts
df_dr_graphics_1 = df_dr[df_dr['graphic']==1]
dr_stat_in_graphic = df_dr_graphics_1[df_dr_graphics_1['statistic']==1]
dr_stat_in_graphic
#so no graphic posts with stat

#look at percentage of videos that were short form
df_dr_videos= df_dr[df_dr['post_type']==3]
df_dr_short_form_vid = df_dr_videos[df_dr_videos['short_form_vid']==1]
dr_number_short_vid= len(df_dr_short_form_vid)
dr_number_videos = len(df_dr_videos)
#calculate ratio
dr_ratio_short_vids = (dr_number_short_vid / dr_number_videos)
print('Ratio of Videos that are short form: ' + str(dr_ratio_short_vids))

#make donut visual for report
#break down short videos vs regular videos
df_dr_long_form_vid = df_dr_videos[df_dr_videos['short_form_vid']==0]
dr_number_short_vid = df_dr_short_form_vid.shape[0]
dr_number_long_vid = df_dr_long_form_vid.shape[0]
dr_total_videos = dr_number_short_vid + dr_number_long_vid
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#4682B4', '#E3D6BF']
wedges, _ = ax.pie([dr_number_short_vid, dr_number_long_vid], labels=['Tik Tok Style', 'Regular Style'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Video Posts: Tik Tok Style vs. Regular Style", fontsize=18)
plt.savefig('Donut Videos tik toks vs regular (DR)')
plt.show()

#calculate median engagement of videos and short videos
#likes
dr_median_likes_short_vid = df_dr_short_form_vid['likes'].median()
#have to create data frame of just long videos
df_dr_regular_vid = df_dr_videos[df_dr_videos['short_form_vid']==0]
dr_median_likes_regular_vid = df_dr_regular_vid['likes'].median()
print('Short form vid median likes: ' + str(dr_median_likes_short_vid))
print('Regular form vid median likes: ' + str(dr_median_likes_regular_vid))
#comments
dr_median_comments_short_vid = df_dr_short_form_vid['comments'].median()
dr_median_comments_regular_vid = df_dr_regular_vid['comments'].median()
print('Short form vid median comments: ' + str(dr_median_comments_short_vid))
print('Regular form vid median comments: ' + str(dr_median_comments_regular_vid))
#understand that since there was just one short form video, it is not the median, its just pulling it

#how many posts had indiviudal quotes
df_dr_indv_quote = df_dr[df_dr['indv_quote']!=0]
print("Number of posts with Indivual Quote featured: " + str(df_dr_indv_quote.shape[0]))
#0 post with individual quotes

#median likes and comments overall
dr_median_likes = df_dr['likes'].median()
print(dr_median_likes)
dr_median_comments = df_dr['comments'].median()
print(dr_median_comments)

#median likes/comments for only graphics posts
dr_median_likes_graphics = df_dr_graphics_1['likes'].median()
print(dr_median_likes_graphics)
dr_median_comments_graphics = df_dr_graphics_1['comments'].median()
print(dr_median_comments_graphics)

##################################################
# WR - World Relief
df_wr.count()

#check how many major news posts there are to see if able to include in heatmap
df_wr_major_news_count= df_wr[df_wr['major_news']==1]
df_wr_major_news_count
#5 major news posts! Enough to put into heatmap but well see if there wil be actual correlations with it

#heatmap
sns.set(style='whitegrid')
plt.figure(figsize=(14,10))
#dropping ID, nonprofit, day and date; will do boolean days
#include bible_quote into heatmap because org
df_wr_without_variables = df_wr.drop(['nonprofit', 'ID', 'date', 'day', 'month'], axis=1)
sns.heatmap(df_wr_without_variables.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('Correlation Matrix Heatmap (WR)', fontsize=18)
plt.savefig('Correlation Matrix Heatmap (WR)')
plt.show
#not enough instances of emojis for heatmap to use it
#likes+major_news(0.57), likes+comments(.81)
#comments+major_news(.43), comments+call_action(.18)

#emojis variable is missing from heatmap,check how many observation points of it there are
df_wr_emojis = df_wr[df_wr['emojis']!=0]
print('Amount of emoji posts used for WRs posts: ' + str(df_wr_emojis.shape[0]))
#no emojis used, explains the faulty plot

#create boolean days to findd potential correlations
df_wr_dummies = pd.get_dummies(df_wr, columns=['day'])
df_wr_dummies.dtypes
df_wr_dummies_without_variables= df_wr_dummies.drop(['nonprofit', 'ID', 'date', 'month'], axis=1)
wr_corr_matrix = df_wr_dummies_without_variables.corr()
wr_like_corr = wr_corr_matrix['likes']
wr_comment_corr = wr_corr_matrix['comments']
print(wr_like_corr)
print(wr_comment_corr)
#potential correlation with fri+likes/comments, sat+likes/comments

#use pearson test to determine statistical significance
# 1) fri/likes (.13)
wr_fri = df_wr_dummies['day_Fri']
wr_likes = df_wr_dummies['likes']
corr_coef, p_value =pearsonr(wr_fri, wr_likes)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
# 2) sat/likes (.13)
wr_sat = df_wr_dummies['day_Sat']
wr_likes = df_wr_dummies['likes']
corr_coef, p_value =pearsonr(wr_sat, wr_likes)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
# 3) fri/comments (.25)
wr_fri = df_wr_dummies['day_Fri']
wr_comments = df_wr_dummies['comments']
corr_coef, p_value =pearsonr(wr_fri, wr_comments)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
# 4) sat/comments (.10)
wr_sat = df_wr_dummies['day_Sat']
corr_coef, p_value =pearsonr(wr_sat, wr_comments)
print("Correlation Coefficient:", corr_coef) 
print("P-value:", p_value)
#only fri/comments has statistical signifiance


#explore correlation leads to engagements/pearsons
#likes+major_news(0.57), likes+comments(.81), comments+major_news(.43), comments+call_action(.18)
wr_likes = df_wr_dummies['likes']
wr_major_news = df_wr_dummies['major_news']
wr_comments = df_wr_dummies['comments']
wr_call_action = df_wr_dummies['call_action']
corr_likes_major_news, p_value_likes_major_news = pearsonr(wr_likes, wr_major_news)
corr_likes_comments, p_value_likes_comments = pearsonr(wr_likes, wr_comments)
corr_comments_major_news, p_value_comments_major_news = pearsonr(wr_comments, wr_major_news)
corr_comments_call_action, p_value_comments_call_action = pearsonr(wr_comments, wr_call_action)
print("Correlation and P-values:")
print(f"Likes vs. Major News: Correlation = {corr_likes_major_news}, P-value = {p_value_likes_major_news}")
print(f"Likes vs. Comments: Correlation = {corr_likes_comments}, P-value = {p_value_likes_comments}")
print(f"Comments vs. Major News: Correlation = {corr_comments_major_news}, P-value = {p_value_comments_major_news}")
print(f"Comments vs. Call to action: Correlation = {corr_comments_call_action}, P-value = {p_value_comments_call_action}")


#explore percentage break down of post_type 1s/2s on DR graphics
#number of graphic photos
df_wr_graphics_1 = df_wr[df_wr['graphic']==1]
print("Number of Graphics on WRs Instagram: " + str(df_wr_graphics_1.shape[0]))
df_wr_graphics_post_type_1 = df_wr_graphics_1[df_wr_graphics_1['post_type']==1]
print("Number of Photos with Graphics on WRs Instagram: " + str(df_wr_graphics_post_type_1.shape[0]))
#now ratio
df_wr_ratio_graphic_post_type_1 = (df_wr_graphics_post_type_1.shape[0] / df_wr_graphics_1.shape[0])
print("Percent of photos that are graphics: " +str(df_wr_ratio_graphic_post_type_1))
#percent of overall posts that are graphics
df_wr_overall_posts_number = df_wr.shape[0]
wr_ratio_graphic_to_overall = ((df_wr_graphics_1.shape[0]) / df_wr_overall_posts_number)
print('Percent of posts that were graphics: ' + str(wr_ratio_graphic_to_overall))
#number of graphic carousels
df_wr_graphic_post_type_2 = df_wr_graphics_1[df_wr_graphics_1['post_type']==2]
print("Number of Carousels with Graphics on WRs Instagram: " + str(df_wr_graphic_post_type_2.shape[0]))
#now ratio
df_wr_ratio_graphic_post_type_2 = (df_wr_graphic_post_type_2.shape[0] / df_wr_graphics_1.shape[0])
print("Percent of carousels that are graphics: " + str(df_wr_ratio_graphic_post_type_2 ))

#make donut for graphic posts split up for report
#data for the donut plot (overall posts vs graphics)
df_wr_graphics_0_3 = df_wr[(df_wr['graphic'] == 0) | (df_wr['graphic'] == 3)]
num_posts_not_graphics = df_wr_graphics_0_3.shape[0]
num_graphics = df_wr_graphics_1.shape[0]
ci_total_posts = num_posts_not_graphics + num_graphics
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFC0CB','#AFEEEE']
wedges, _ = ax.pie([num_graphics, num_posts_not_graphics], labels=['Graphics', 'Not/Other Post'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Share of Graphics among Posts", fontsize=18)
plt.savefig('Donut Graphics vs overall posts (WR)')
plt.show()

#make donut for graphic posts split up for report
# Data for the donut plot
num_photos = df_wr_graphics_post_type_1.shape[0]
num_carousels = df_wr_graphic_post_type_2.shape[0]
total_graphics = num_photos + num_carousels
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#90EE90','#F08080']
wedges, _ = ax.pie([num_photos, num_carousels], labels=['Photos', 'Carousels'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Photos vs. Carousels", fontsize=18)
plt.savefig('Donut Graphics carousels vs photos (WR)')
plt.show()

#explore percentage break down of persons_needs and org_persons in graphic images
#number of graphic with images
df_wr_graphic_image_1 = df_wr[df_wr['graphic_with_image']==1]
print("Number of times there is a Graphic with an Image on WRs Instagram: " + str(df_wr_graphic_image_1.shape[0]))
#number of persons_need in graphic images
wr_persons_need_1 = df_wr_graphic_image_1[df_wr_graphic_image_1['persons_in_need']==1].shape[0]
print("Number of times there is a graphic with an image of persons in need: " + str(wr_persons_need_1))
#ratio 
wr_ratio_persons_need_in_graphic = (wr_persons_need_1 / df_wr_graphic_image_1.shape[0])
print("Ratio of Persons in Need in Graphics: " + str(wr_ratio_persons_need_in_graphic))
#number of org_persons in graphic images
wr_org_persons_1 = df_wr_graphic_image_1[df_wr_graphic_image_1['org_persons']==1].shape[0]
print("Number of times there is a graphic with Org Persons: " + str(wr_org_persons_1))
#percentage of graphics that had images in them
wr_graphic_images_to_graphic = (df_wr_graphic_image_1.shape[0] / df_wr_graphics_1.shape[0])
print("Percentage of the Graphics had images: " + str(wr_graphic_images_to_graphic))

#make donut for graphic with image breakdown report
#Data for the donut plot (graphics vs graphics with images)
wr_number_graphic_image = df_wr_graphic_image_1.shape[0]
df_wr_number_not_image = df_wr_graphics_1[df_wr_graphics_1['graphic_with_image']==0]
wr_number_graphic_not_image = df_wr_number_not_image.shape[0]
total_graphics = wr_number_graphic_not_image + wr_number_graphic_image
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#B03060','#40E0D0']
wedges, _ = ax.pie([wr_number_graphic_not_image, wr_number_graphic_image], labels=['No Image', 'Image'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Graphic Posts: Real Photo Image Included", fontsize=18)
plt.savefig('Donut Graphics image vs not (WR)')
plt.show()

#explore percentage breakdown of of persons_needs and org_persons in per_story for CI
#number of personal stories
df_wr_per_story_1 = df_wr[df_wr['per_story']==1]
print("Number of Personal Story Posts on WRs Instagram: " + str(df_wr_per_story_1.shape[0]))
#number of persons_needs personal stories
df_wr_persons_needs_per_story = df_wr_per_story_1[df_wr_per_story_1['persons_in_need']==1]
print("Number of Person in Need's Personal Story Posts: " + str(df_wr_persons_needs_per_story.shape[0]))
#Ratio
df_wr_ratio_persons_needs_story = (df_wr_persons_needs_per_story.shape[0] / df_wr_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_wr_ratio_persons_needs_story)) 
#number of org_persons personal stories
df_wr_org_persons_per_story = df_wr_per_story_1[df_wr_per_story_1['org_persons']==1]
print("Number of Org Persons's Personal Story Posts: " + str(df_wr_org_persons_per_story.shape[0]))
#Ratio
df_wr_ratio_org_persons_story = (df_wr_org_persons_per_story.shape[0] / df_wr_per_story_1.shape[0])
print("Percetange of Personal Stories that are Persons in Need: " + str(df_wr_ratio_org_persons_story))
#Ratio of personal story to overall posts
df_wr_overall_posts_number = df_wr.shape[0]
wr_ratio_per_story_to_overall = ((df_wr_per_story_1.shape[0]) / df_wr_overall_posts_number)
print('Percentage of overall posts that are personal stories: ' + str(wr_ratio_per_story_to_overall))

#make donut for personal story posts vs overall posts
#Data for the donut plot
df_wr_post_not_per = df_wr[df_wr['per_story']==0]
wr_overall_posts_not_personal_story = df_wr_post_not_per.shape[0]
wr_number_per_story = df_wr_per_story_1.shape[0]
total_graphics = wr_overall_posts_not_personal_story + wr_number_per_story
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#FFA500','#0000FF']
wedges, _ = ax.pie([wr_overall_posts_not_personal_story, wr_number_per_story], labels=['Not/Other Post', 'Personal Story'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Posts: Personal Story Posts", fontsize=18)
plt.savefig('Donut Presonal Story vs overall (WR)')
plt.show()

#find median enagement of personal story posts vs not
wr_median_likes_not_story = df_wr_post_not_per['likes'].median()
wr_median_likes_story = df_wr_per_story_1['likes'].median()
print('Not personal story median likes: ' + str(wr_median_likes_not_story))
print('Personal Story median likes: ' + str(wr_median_likes_story))
#comments
wr_median_comments_not_story= df_wr_post_not_per['comments'].median()
wr_median_comments_story = df_wr_per_story_1['comments'].median()
print('Not personal story median comments: ' + str(wr_median_comments_not_story))
print('Personal Story median comments: ' + str(wr_median_comments_story ))


#distribution of WR posts per day
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.countplot(x='day', data=df_wr, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Post by Day Posted (WR)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Distribution of Posts Analyzed by Day Posted (WR)')
plt.show()

#donut of DR distribution of 3 different post types
df_wr_post_type_count = df_wr['post_type'].value_counts()
df_wr_post_type_count
labels = ['Photo', 'Video', 'Carousel']
colors = {'Carousel': 'skyblue', 'Video': 'lightcoral', 'Photo': 'lightgreen'}
wr_sizes = df_wr_post_type_count.values
plt.figure(figsize=(10, 6))
wedges, _, autotexts = plt.pie(wr_sizes, labels=labels, colors=[colors[label] for label in labels], textprops={'fontsize':16},
                               autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(wr_sizes))})', startangle=90, wedgeprops=dict(width=0.8, edgecolor='w'))
#appearance of the text
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_color('black')
#white circle at the center to make it a donut plot
inner_circle = plt.Circle((0, 0), 0.4, color='white')
plt.gca().add_artist(inner_circle)
plt.title('Distribution of Post Types (WR)', fontsize=20)
plt.axis('equal')
plt.savefig('Donut Post Types for WR')
plt.show()

#now looking at comparison of likes/comments across days and post_types
#Creating median likes for post_types plot
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue', 'Video': 'lightcoral'}
wr_median_likes_per_post_type = df_wr.groupby('post_type')['likes'].median()
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_wr['post_type'], y=df_wr['likes'], palette=colors_dict.values())
#print median value on each box
medians = df_wr.groupby('day')['likes'].median()
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Likes Count', fontsize=16)
plt.title('Median Likes for Each Post Type (WR)', fontsize=20)
plt.ylim(0,150)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Likes for Post Type (WR)')
plt.show()

#Creating median comments for post_types plot
colors_dict = {'Photo': 'lightgreen', 'Carousel': 'skyblue', 'Video': 'lightcoral'}
wr_median_comments_per_post_type = df_wr.groupby('post_type')['comments'].median()
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_wr['post_type'], y=df_wr['comments'], palette=colors_dict.values())
#print median value on each box
medians = df_wr.groupby('day')['comments'].median()
post_type_names = ['Photo', 'Carousel', 'Video']
plt.xticks(range(len(post_type_names)), post_type_names)
plt.xlabel('Post Type', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.ylabel('Median Comments Count', fontsize=16)
plt.title('Median Comments for Each Post Type (WR)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Median Comments for Post Type (WR)')
plt.ylim(0,8)
plt.show()

#calculate/graph median likes to best days, will use later to compare to other nonprofits
wr_median_likes_per_day = df_wr.groupby('day')['likes'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_wr['day'], y=df_wr['likes'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_wr.groupby('day')['likes'].median()
for idx, day in enumerate(wr_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, wr_median_likes_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Median Likes Count', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=14)
plt.title('Median Likes for Each Day (WR)', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,200)
plt.savefig('Median Likes for Each Day(WR)')
plt.show()

#do the same for median comments
wr_median_comments_per_day = df_wr.groupby('day')['comments'].median().reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")
ax = sns.boxplot(x=df_wr['day'], y=df_wr['comments'], order=['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'], palette=colors)
#print median value on each box
medians = df_wr.groupby('day')['comments'].median()
for idx, day in enumerate(wr_median_likes_per_day.index):
    ax.annotate(f'{medians[day]}', (idx, wr_median_comments_per_day[day]), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Median Comments Count', fontsize=14)
plt.title('Median Comments for Each Day (WR)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0,6)
plt.savefig('Median Comments for Each Day(WR)')
plt.show()

#check amount of posts that used emojis/hashtags to see if enough for violin plots
df_wr_hashtags = df_wr[df_wr['hashtags']!=0]
print('Amount of hashtags used for WRs posts: ' + str(df_wr_hashtags.shape[0]))
df_wr_emojis = df_wr[df_wr['emojis']!=0]
print('Amount of hashtags used for WRs posts: ' + str(df_wr_emojis.shape[0]))
#no emojis used but 59 posts with hashtags

#violin plot for hashtags by post type
wr_hashtags_by_post= df_wr.groupby('post_type')['hashtags'].mean()
plt.figure(figsize=(10,6))
colors = ["lightgreen", "skyblue", "lightcoral"]
sns.violinplot(x='post_type', y='hashtags', data=df_wr, palette=colors)
plt.xlabel('Post Type', fontsize=16)
plt.ylabel('Average Hashtags', fontsize=16)
#to avoid type error
post_type_labels = ['Photo', 'Carousel', 'Video']
plt.xticks(ticks=[0, 1, 2], labels=post_type_labels, fontsize=14)
plt.title('Usage of Hashtags by Post Type (WR)', fontsize=20)
plt.tight_layout()
plt.ylim((0,12))
plt.savefig('Violin hashtags by post type (WR)')
plt.show()

#looking at percentage of graphics that have a statistic present
df_wr_statistic_1 = df_wr[df_wr['statistic']==1]
df_wr_statistic_1
#5 statistic posts
df_wr_graphics_1 = df_wr[df_wr['graphic']==1]
wr_stat_in_graphic = df_wr_graphics_1[df_wr_graphics_1['statistic']==1]
wr_stat_in_graphic
#4 were in a graphic

#look at percentage of videos that were short form
df_wr_videos= df_wr[df_wr['post_type']==3]
df_wr_short_form_vid = df_wr_videos[df_wr_videos['short_form_vid']==1]
wr_number_short_vid= len(df_wr_short_form_vid)
wr_number_videos = len(df_wr_videos)
#calculate ratio
wr_ratio_short_vids = (wr_number_short_vid / wr_number_videos)
print('Ratio of Videos that are short form: ' + str(wr_ratio_short_vids))

#make donut visual for report
#break down short videos vs regular videos
df_wr_long_form_vid = df_wr_videos[df_wr_videos['short_form_vid']==0]
wr_number_short_vid = df_wr_short_form_vid.shape[0]
wr_number_long_vid = df_wr_long_form_vid.shape[0]
wr_total_videos = wr_number_short_vid + wr_number_long_vid
fig, ax = plt.subplots()
ax.axis('equal')
colors = ['#4682B4', '#E3D6BF']
wedges, _ = ax.pie([wr_number_short_vid, wr_number_long_vid], labels=['Tik Tok Style', 'Regular Style'], colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
center_circle = plt.Circle((0, 0), 0.25, color='white')
ax.add_artist(center_circle)
ax.set_title("Video Posts: Tik Tok Style vs. Regular Style", fontsize=18)
plt.savefig('Donut Videos tik toks vs regular (WR)')
plt.show()

#calculate median engagement of videos and short videos
#likes
wr_median_likes_short_vid = df_wr_short_form_vid['likes'].median()
#have to create data frame of just long videos
df_wr_regular_vid = df_wr_videos[df_wr_videos['short_form_vid']==0]
wr_median_likes_regular_vid = df_wr_regular_vid['likes'].median()
print('Short form vid median likes: ' + str(wr_median_likes_short_vid))
print('Regular form vid median likes: ' + str(wr_median_likes_regular_vid))
#comments
wr_median_comments_short_vid = df_wr_short_form_vid['comments'].median()
wr_median_comments_regular_vid = df_wr_regular_vid['comments'].median()
print('Short form vid median comments: ' + str(wr_median_comments_short_vid))
print('Regular form vid median comments: ' + str(wr_median_comments_regular_vid))
#understand that since there was just one short form video, it is not the median, its just pulling it

#how many posts had indiviudal quotes
df_wr_indv_quote = df_wr[df_wr['indv_quote']!=0]
print("Number of posts with Indivual Quote featured: " + str(df_wr_indv_quote.shape[0]))
#3 posts with individual quotes

#median likes and comments overall
wr_median_likes = df_wr['likes'].median()
print(wr_median_likes)
wr_median_comments = df_wr['comments'].median()
print(wr_median_comments)

#median likes/comments for only graphics posts
wr_median_likes_graphics = df_wr_graphics_1['likes'].median()
print(wr_median_likes_graphics)
wr_median_comments_graphics = df_wr_graphics_1['comments'].median()
print(wr_median_comments_graphics)

#median likes/comments for only non graphics posts
wr_median_likes_non_graphics = df_wr_graphics_0_3['likes'].median()
print(wr_median_likes_non_graphics)
wr_median_comments_non_graphics = df_wr_graphics_0_3['comments'].median()
print(wr_median_comments_non_graphics)

#calculate median engagement of graphic photos and carousels
#likes
wr_median_likes_graphic_photo = df_wr_graphics_post_type_1['likes'].median()
wr_median_likes_graphic_carousel = df_wr_graphic_post_type_2['likes'].median()
print('Graphic photo median likes: ' + str(wr_median_likes_graphic_photo))
print('Graphic carousel median likes: ' + str(wr_median_likes_graphic_carousel))
#comments
wr_median_comments_graphic_photo = df_wr_graphics_post_type_1['comments'].median()
wr_median_comments_graphic_carousel = df_wr_graphic_post_type_2['comments'].median()
print('Graphic photo median comments: ' + str(wr_median_comments_graphic_photo))
print('Graphic carousel median comments: ' + str(wr_median_comments_graphic_carousel))

#find median enagement of personal story posts vs not
wr_median_likes_not_story = df_wr_post_not_per['likes'].median()
wr_median_likes_story = df_wr_per_story_1['likes'].median()
print('Not personal story median likes: ' + str(wr_median_likes_not_story))
print('Personal Story median likes: ' + str(wr_median_likes_story))
#comments
wr_median_comments_not_story= df_wr_post_not_per['comments'].median()
wr_median_comments_story = df_wr_per_story_1['comments'].median()
print('Not personal story median comments: ' + str(wr_median_comments_not_story))
print('Personal Story median comments: ' + str(wr_median_comments_story ))

#calculate median engagement of videos and short videos
#likes
wr_median_likes_short_vid = df_wr_short_form_vid['likes'].median()
#have to create data frame of just long videos
df_wr_regular_vid = df_wr_videos[df_wr_videos['short_form_vid']==0]
wr_median_likes_regular_vid = df_wr_regular_vid['likes'].median()
print('Short form vid median likes: ' + str(wr_median_likes_short_vid))
print('Regular form vid median likes: ' + str(wr_median_likes_regular_vid))
#comments
wr_median_comments_short_vid = df_wr_short_form_vid['comments'].median()
wr_median_comments_regular_vid = df_wr_regular_vid['comments'].median()
print('Short form vid median comments: ' + str(wr_median_comments_short_vid))
print('Regular form vid median comments: ' + str(wr_median_comments_regular_vid))
