"""Add us_code_section table

Revision ID: 2fd7b3c22451
Revises: 29a85ca87e53
Create Date: 2025-04-07 12:23:34.174190

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2fd7b3c22451'
down_revision: Union[str, None] = '29a85ca87e53'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('us_code_section',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('parent_id', sa.Integer(), nullable=True),
    sa.Column('title_number', sa.Integer(), nullable=True),
    sa.Column('subtitle', sa.Text(), nullable=True),
    sa.Column('chapter', sa.Text(), nullable=True),
    sa.Column('subchapter', sa.Text(), nullable=True),
    sa.Column('part', sa.Text(), nullable=True),
    sa.Column('subpart', sa.Text(), nullable=True),
    sa.Column('section_number', sa.Text(), nullable=False),
    sa.Column('section_title', sa.Text(), nullable=True),
    sa.Column('full_text', sa.Text(), nullable=True),
    sa.Column('amendment_count', sa.Integer(), nullable=False),
    sa.Column('special_formatting', sa.Boolean(), nullable=False),
    sa.Column('updated_at', sa.Date(), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['parent_id'], ['us_code_section.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('us_code_section')
    # ### end Alembic commands ###
