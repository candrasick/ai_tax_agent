"""Add unique constraint to section_number in us_code_section

Revision ID: 378a4bbf05e9
Revises: 85b6a9c8d02f
Create Date: 2025-04-10 10:49:20.962464

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '378a4bbf05e9'
down_revision: Union[str, None] = '85b6a9c8d02f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('us_code_section', schema=None) as batch_op:
        batch_op.create_unique_constraint('uq_section_number', ['section_number'])
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('us_code_section', schema=None) as batch_op:
        batch_op.drop_constraint('uq_section_number', type_='unique')
    # ### end Alembic commands ###
