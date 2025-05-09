"""add section_history table and revised_financial_impact column

Revision ID: 4646bdd2f0d3
Revises: 369c83f5e6e4
Create Date: 2025-04-18 14:56:26.253975

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4646bdd2f0d3'
down_revision: Union[str, None] = '369c83f5e6e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('section_history',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('orig_section_id', sa.Integer(), nullable=False),
    sa.Column('version_changed', sa.Integer(), nullable=False, comment='Version number where this change occurred'),
    sa.Column('action', sa.Text(), nullable=False, comment='Action taken (e.g., simplify, delete, keep, redraft)'),
    sa.Column('rationale', sa.Text(), nullable=True, comment='Rationale for the action'),
    sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['orig_section_id'], ['us_code_section.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_section_history_orig_section_id'), 'section_history', ['orig_section_id'], unique=False)
    op.create_index(op.f('ix_section_history_version_changed'), 'section_history', ['version_changed'], unique=False)
    op.add_column('us_code_section_revised', sa.Column('revised_financial_impact', sa.Numeric(), nullable=True, comment='Estimated financial impact of this revised version.'))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('us_code_section_revised', 'revised_financial_impact')
    op.drop_index(op.f('ix_section_history_version_changed'), table_name='section_history')
    op.drop_index(op.f('ix_section_history_orig_section_id'), table_name='section_history')
    op.drop_table('section_history')
    # ### end Alembic commands ###
