#%%
# Load data
#

#%% ===========================================================================
# Data source and paths
# =============================================================================
path_data = Path(PATH_DATA_ROOT, r"").expanduser()
assert path_data.exists(), "Data path does not exist: {}".format(path_data)
logging.info("Data path {}".format(PATH_DATA_ROOT))

#%% ===========================================================================
# Load data
# =============================================================================
logging.info(f"Loading files into memory")

path_labels = path_data / "annotations" / "list.txt"
assert path_labels.exists()

df = pd.read_csv(path_labels, header=6, delim_whitespace=True)
df.columns = ['image file name', 'class', 'species', 'breed']
df['species'] = df['species'].astype('category')
df['species'].cat.rename_categories({1:'Cat', 2:'Dog'}, inplace=True)

# Get the breed of the animal from the file name
def split_breed(x):
    breed_strings = x.split('_')
    breed_strings.pop()
    return " ".join(breed_strings)

df['breed string'] = df['image file name'].apply(split_breed)

# %%
# Collect the paths of the data augmentations

# Utilities
def file_path(x, path_data, file_pattern):
    filename = file_pattern.format(x)
    return path_data / filename

def assert_path_exists(x):
    return x.exists()

# Define the full file path for the images
file_path_parameterized = functools.partial(file_path, path_data=path_data/'images', file_pattern='{}.jpg')
df['path_image'] = df['image file name'].apply(file_path_parameterized)
assert df['path_image'][0].exists()

# Define the xml path
path_xml = path_data / 'annotations' / 'xmls'
file_path_parameterized = functools.partial(file_path, path_data=path_xml, file_pattern='{}.xml')
df['path_xml'] = df['image file name'].apply(file_path_parameterized)
assert df['path_xml'][0].exists()

# Define the trimap path
path_trimap = path_data / 'annotations' / 'trimaps'
file_path_parameterized = functools.partial(file_path, path_data=path_trimap, file_pattern='._{}.png')
df['path_trimap'] = df['image file name'].apply(file_path_parameterized)
assert df['path_trimap'][0].exists()

# Check paths
# assert df['path_xml'].apply(assert_path_exists).all()
# assert df['path_image'].apply(assert_path_exists).all()
# assert df['path_xml'][0].exists()

#%% XML expansion

# Load XML to dictionary
def load_xml(x):
    # print(x)
    if x.exists():
        with open(x) as fd:
            doc = xmltodict.parse(fd.read())
        return doc
    else:
        return 0
assert df['path_trimap'][0].exists()
# file_path_parameterized = functools.partial(file_path, path_data=path_trimap, file_pattern='._{}.png')
df['xml_dict'] = df['path_xml'].apply(load_xml)

#%%
# Expand dictionary to columns
def collect_annotations(x):
    if x:
        try:
            x['image_width'] = x['annotation']['size']['width']
            x['image_height'] = x['annotation']['size']['height']
            x['image_pose'] = x['annotation']['object']['pose']
            x['image_truncated'] = x['annotation']['object']['truncated']
            x['image_occluded'] = x['annotation']['object']['occluded']
            x['image_difficult'] = x['annotation']['object']['difficult']
            x['image_bounding_box'] = x['annotation']['object']['bndbox']
            return x
        except:
            return 0
    else:
        return 0

# df2 = df['xml_dict'].apply(collect_annotations, result_type='expand')
applied_df = df.apply(lambda row: collect_annotations(row['xml_dict']), axis='columns', result_type='expand')
sum(applied_df['annotation'] != 0)

#
# df['image_size'] = df['xml_dict'].apply(lambda x: x['annotation']['size'] if x)
# df['image_height'] = df['xml_dict'][0]['annotation']['size']['height']
# df[]
# ('size',
#  OrderedDict([('width', '450'),
#               ('height', '313'),
#               ('depth', '3')])),
# %%
class AIDataSet():
    def __init__(self, image_folder, image_extension, df, col_file, col_target):
        """

        :param image_folder:
        :param image_extension:
        :param df:
        :param col_target:
        """
        self.image_folder = image_folder
        self.image_extension = image_extension
        self.df = df
        self.col_file = col_file
        self.col_target = col_target

        logging.info("{} images found".format(len(self.list_images())))
        logging.info("Dataframe with {} records, target column is {}".format(len(df), self.col_target))
        logging.info("{} classes".format(len(self.df[self.col_target].value_counts())))
        logging.info("Classes counts: {}".format(list(self.df[self.col_target].value_counts().iteritems())))

        self.check_alignment(strict=False)

    def list_images(self):
        image_list = self.image_folder.glob("*"+self.image_extension)
        return list(image_list)

    def check_alignment(self, strict):
        # Get the file names from the image folder
        image_names = [p.stem for p in self.list_images()]
        # Compare the number of images in the dataframe to the images in folder
        extra_df_records = self.df[self.df['file name'].isin(image_names) == False]
        logging.info("{} extra images in the dataframe".format(len(extra_df_records)))

        # Compare the number of images in the folder to the dataframe labels
        extra_folder_images = pd.Series(image_names)[pd.Series(image_names).isin(df['file name']) == False]
        logging.info("{} extra images found in the folder".format(len(extra_folder_images)))
        if strict:
            assert extra_df_records == 0
            assert extra_folder_images == 0

class Plotter():
    def plot12(self, dataset, ts_string_indices, source_jpg_folder='jpg_images', extension='jpg', rows=3, cols=4):
        pass

# %%
image_folder = path_data / "images"
image_extension = '.jpg'
ds = AIDataSet(image_folder, image_extension, df, 'file name', 'species')
df.head()

# %% Plotting
ROWS = 3
COLS = 5
NUM_IMAGES = ROWS * COLS

# Figure ##############################################################
# figsize = [width, height]
fig = plt.figure(figsize=PAPER["A3_LANDSCAPE"], facecolor='white')
fig.suptitle("Sample images".format(), fontsize=20)

plot_indices = df.sample(NUM_IMAGES)['file name'].values

for i in range(NUM_IMAGES):
    record = df.sample(1)
    file_stem = record['file name'].tolist()[0]
    label = '\n'.join(record[['species', 'breed string']].values.tolist()[0])
    ax = fig.add_subplot(ROWS, COLS, i + 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    selected_image_path = list(image_folder.glob(file_stem + image_extension))
    assert len(selected_image_path) == 1
    selected_image_path = selected_image_path.pop()
    assert selected_image_path.exists()
    img = mpl.image.imread(selected_image_path)
    ax.imshow(img)
    ax.set_title(file_stem)

    t = ax.text(10, 50, label, color='black', alpha=1)
    t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
plt.show()