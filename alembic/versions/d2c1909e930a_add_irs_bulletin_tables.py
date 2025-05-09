"""Add IRS bulletin tables

Revision ID: d2c1909e930a
Revises: 6b6fbb6db86c
Create Date: 2025-04-09 11:50:28.523956

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd2c1909e930a'
down_revision: Union[str, None] = '6b6fbb6db86c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('irs_bulletin',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('bulletin_number', sa.Text(), nullable=False),
    sa.Column('bulletin_date', sa.Date(), nullable=True),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('source_url', sa.Text(), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('irs_bulletin_item',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('bulletin_id', sa.Integer(), nullable=False),
    sa.Column('item_type', sa.Text(), nullable=True),
    sa.Column('item_number', sa.Text(), nullable=False),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('action', sa.Text(), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('full_text', sa.Text(), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['bulletin_id'], ['irs_bulletin.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('irs_bulletin_item_to_code_section',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('bulletin_item_id', sa.Integer(), nullable=False),
    sa.Column('section_id', sa.Integer(), nullable=False),
    sa.Column('relevance_notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['bulletin_item_id'], ['irs_bulletin_item.id'], ),
    sa.ForeignKeyConstraint(['section_id'], ['us_code_section.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('bulletin_item_id', 'section_id', name='uq_bulletin_item_section')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('irs_bulletin_item_to_code_section')
    op.drop_table('irs_bulletin_item')
    op.drop_table('irs_bulletin')
    # ### end Alembic commands ###
