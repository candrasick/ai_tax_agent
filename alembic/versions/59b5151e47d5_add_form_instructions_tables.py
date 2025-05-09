"""add_form_instructions_tables

Revision ID: 59b5151e47d5
Revises: 82753e3699de
Create Date: 2025-04-10 09:47:11.739685

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '59b5151e47d5'
down_revision: Union[str, None] = '82753e3699de'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('form_instructions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=255), nullable=False),
    sa.Column('form_number', sa.String(length=50), nullable=False),
    sa.Column('html_url', sa.String(length=255), nullable=True),
    sa.Column('pdf_url', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_form_instructions_form_number'), 'form_instructions', ['form_number'], unique=False)
    op.create_table('form_fields',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('instruction_id', sa.Integer(), nullable=False),
    sa.Column('field_label', sa.String(length=100), nullable=False),
    sa.Column('full_text', sa.Text(), nullable=False),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['instruction_id'], ['form_instructions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('form_fields')
    op.drop_index(op.f('ix_form_instructions_form_number'), table_name='form_instructions')
    op.drop_table('form_instructions')
    # ### end Alembic commands ###
