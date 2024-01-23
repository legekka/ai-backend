from peewee import *
from modules.config import Config

db = PostgresqlDatabase(Config.db_name, user=Config.db_user, password=Config.db_password,
                        host=Config.db_host, port=Config.db_port)

class BaseModel(Model):
    class Meta:
        database = db

class Image(BaseModel):
    id = BigAutoField(primary_key=True)
    filename = CharField(max_length=256, unique=True)
    image_512_t = TextField()
    image_768 = TextField()
    image_305 = TextField()

    sankaku_id = BigIntegerField(null=True) # Sankaku post ID, this is part of the TODO training database implementation, also used for MontagepostImages optionally

    class Meta:
        database = db
        table_name = 'image'

class Tag(BaseModel):
    id = BigAutoField(primary_key=True)
    name = CharField(max_length=256, unique=True)

    class Meta:
        database = db
        table_name = 'tag'

class User(BaseModel):
    id = BigAutoField(primary_key=True)
    username = CharField(max_length=45, unique=True)
    discord_id = BigIntegerField(unique=True) 

    class Meta:
        database = db
        table_name = 'user'

class ImageTag(BaseModel):
    id = BigAutoField(primary_key=True)
    image_id = ForeignKeyField(Image, backref='image_tags')
    tag_id = ForeignKeyField(Tag, backref='tag_images')

    class Meta:
        database = db
        table_name = 'image_tag'

class Rating(BaseModel):
    id = BigAutoField(primary_key=True)
    image_id = ForeignKeyField(Image, backref='image_ratings')
    user_id = ForeignKeyField(User, backref='user_ratings')
    rating = DecimalField(decimal_places=2, max_digits=3, choices=[(0, '0'), (0.17, '0.17'), (0.33, '0.33'), (0.5, '0.5'), (0.67, '0.67'), (0.83, '0.83'), (1, '1')])

    class Meta:
        database = db
        table_name = 'rating'

class Montagepost(BaseModel):
    id = BigAutoField(primary_key=True)
    user_id = ForeignKeyField(User, backref='user_montageposts')
    created_at = DateTimeField()

    class Meta:
        database = db
        table_name = 'montagepost'

class MontagepostImage(BaseModel):
    id = BigAutoField(primary_key=True)
    montagepost_id = ForeignKeyField(Montagepost, backref='montagepost_images')
    image_id = ForeignKeyField(Image, backref='image_montageposts')

    class Meta:
        database = db
        table_name = 'montagepost_image'