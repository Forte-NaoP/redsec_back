"""MODEL_uuid

Revision ID: 6c1f318bb439
Revises: 42142315a416
Create Date: 2023-09-18 20:38:10.421203

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6c1f318bb439'
down_revision: Union[str, None] = '42142315a416'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('ML_model', schema=None) as batch_op:
        batch_op.add_column(sa.Column('ML_model_name', sa.String(), nullable=False))
        batch_op.add_column(sa.Column('ML_model_uuid', sa.String(), nullable=False))
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=False))
        batch_op.drop_constraint('uq_model_model_name', type_='unique')
        batch_op.create_unique_constraint(batch_op.f('uq_ML_model_ML_model_uuid'), ['ML_model_uuid'])
        batch_op.create_foreign_key(batch_op.f('fk_ML_model_user_id_user'), 'user', ['user_id'], ['id'])
        batch_op.drop_column('model_path')
        batch_op.drop_column('model_name')
        batch_op.drop_column('model_pubkey')

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('ML_model', schema=None) as batch_op:
        batch_op.add_column(sa.Column('model_pubkey', sa.VARCHAR(), nullable=False))
        batch_op.add_column(sa.Column('model_name', sa.VARCHAR(), nullable=False))
        batch_op.add_column(sa.Column('model_path', sa.VARCHAR(), nullable=False))
        batch_op.drop_constraint(batch_op.f('fk_ML_model_user_id_user'), type_='foreignkey')
        batch_op.drop_constraint(batch_op.f('uq_ML_model_ML_model_uuid'), type_='unique')
        batch_op.create_unique_constraint('uq_model_model_name', ['model_name'])
        batch_op.drop_column('user_id')
        batch_op.drop_column('ML_model_uuid')
        batch_op.drop_column('ML_model_name')

    # ### end Alembic commands ###
