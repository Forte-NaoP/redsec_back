"""ML_model to model

Revision ID: 1fb25a9bee13
Revises: 6c1f318bb439
Create Date: 2023-09-18 20:47:50.395622

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1fb25a9bee13'
down_revision: Union[str, None] = '6c1f318bb439'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('model',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ML_model_name', sa.String(), nullable=False),
    sa.Column('ML_model_uuid', sa.String(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name=op.f('fk_model_user_id_user')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model')),
    sa.UniqueConstraint('ML_model_uuid', name=op.f('uq_model_ML_model_uuid'))
    )
    op.drop_table('ML_model')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('ML_model',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('ML_model_name', sa.VARCHAR(), nullable=False),
    sa.Column('ML_model_uuid', sa.VARCHAR(), nullable=False),
    sa.Column('user_id', sa.INTEGER(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id', name='pk_model'),
    sa.UniqueConstraint('ML_model_uuid', name='uq_ML_model_ML_model_uuid')
    )
    op.drop_table('model')
    # ### end Alembic commands ###
