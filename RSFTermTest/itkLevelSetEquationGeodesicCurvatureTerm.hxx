/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __itkLevelSetEquationGeodesicCurvatureTerm_hxx
#define __itkLevelSetEquationGeodesicCurvatureTerm_hxx

#include "itkLevelSetEquationGeodesicCurvatureTerm.h"

namespace itk
{
template< class TInput, class TLevelSetContainer >
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::LevelSetEquationGeodesicCurvatureTerm()
{
  for( unsigned int i = 0; i < ImageDimension; i++ )
    {
    this->m_NeighborhoodScales[i] = 1.0;
    }
  this->m_TermName = "Geodesic Curvature Term";
  this->m_RequiredData.insert( "MeanCurvature" );
}

template< class TInput, class TLevelSetContainer >
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::~LevelSetEquationGeodesicCurvatureTerm()
{
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::Value( const LevelSetInputIndexType& iP, const LevelSetDataType& iData )
{
  // MeanCurvature has should be computed by this point.
  itkAssertInDebugAndIgnoreInReleaseMacro( iData.MeanCurvature.m_Computed == true );

  if ( m_GeodesicCurvatureOn )
    {
    return iData.MeanCurvature.m_Value * m_PotentialImage->GetPixel(iP);
    }
  else
    {
    return iData.MeanCurvature.m_Value;
    }
}

template< class TInput, class TLevelSetContainer >
void
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::InitializeParameters()
{
  this->m_CurrentLevelSet=InputImageType::New();
  GenerateImage( this->m_CurrentLevelSet );

  this->m_CurrentHeaviside=InputImageType::New();
  GenerateImage( this->m_CurrentHeaviside );

  currentIteration=0;
  this->SetUp();
}

template< class TInput, class TLevelSetContainer >
void
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::Initialize( const LevelSetInputIndexType& )
{
}

template< class TInput, class TLevelSetContainer >
void
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::Update()
{

}

template< class TInput, class TLevelSetContainer >
void
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::UpdatePixel( const LevelSetInputIndexType& itkNotUsed( iP ),
               const LevelSetOutputRealType& itkNotUsed( oldValue ),
               const LevelSetOutputRealType& itkNotUsed( newValue ) )
{
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::Value( const LevelSetInputIndexType& iP )
{
  if (  m_GeodesicCurvatureOn)
    {
    return this->m_CurrentLevelSetPointer->EvaluateMeanCurvature( iP )*m_PotentialImage->GetPixel(iP);
    }
  else
    {
    return this->m_CurrentLevelSetPointer->EvaluateMeanCurvature( iP );
    }

}
template< class TInput, class TLevelSetContainer >
void LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::GetCurrentHeavisideImage()
{
  typename itk::ImageRegionIterator<  InputImageType > it( m_CurrentHeaviside, m_CurrentHeaviside->GetBufferedRegion() );
  it.GoToBegin();
  typename itk::ImageRegionIterator<  InputImageType > itInverse( this->m_CurrentHeavisideInverse, this->m_CurrentHeavisideInverse->GetBufferedRegion() );
  itInverse.GoToBegin();
  typename itk::ImageRegionIterator<  InputImageType > itLevel( m_CurrentLevelSet, m_CurrentLevelSet->GetBufferedRegion() );
  itLevel.GoToBegin();

  while( !it.IsAtEnd()&& !itInverse.IsAtEnd())
    {
    const LevelSetOutputRealType value =static_cast< LevelSetOutputRealType >( this->m_CurrentLevelSetPointer->Evaluate( it.GetIndex()  ) );
    itLevel.Set(value);
    ++itLevel;
    //LevelSetPointer levelSet = this->m_LevelSetContainer->GetLevelSet( this->m_CurrentLevelSetId );
    //LevelSetOutputRealType value = levelSet->Evaluate( it.GetIndex()  );
    const LevelSetOutputRealType d_val1 = this->m_Heaviside->Evaluate( -value );
    it.Set( d_val1);
    ++it;
    const LevelSetOutputRealType d_val2 = this->m_Heaviside->Evaluate( value );
    itInverse.Set( d_val2);
    ++itInverse;
    }
}
template< class TInput, class TLevelSetContainer >
void LevelSetEquationGeodesicCurvatureTerm< TInput, TLevelSetContainer >
::GenerateImage(InputImagePointer ioImage )
{
  typename InputImageType::IndexType  index;
  index.Fill( 0 );
  std::vector<float> imgExt(3, 0);
  imgExt[0] = this->m_Input->GetBufferedRegion().GetSize()[0];
  imgExt[1] = this->m_Input->GetBufferedRegion().GetSize()[1];
  imgExt[2] = this->m_Input->GetBufferedRegion().GetSize()[2];
  typename InputImageType::SizeType   size;
  size[0]=imgExt[0];
  size[1]=imgExt[1];
  size[2]=imgExt[2];
  typename InputImageType::RegionType region;
  region.SetIndex( index );
  region.SetSize( size );
  typename InputImageType::SpacingType spacing;
  spacing=this->m_Input->GetSpacing();
  // spacing[1]=m_Input->GetSpacing()[1];
  //spacing[2]=m_Input->GetSpacing()[2];
  typename InputImageType::PointType origin;
  origin=this->m_Input->GetOrigin();
  //origin[1]=m_Input->GetOrigin()[1];
  //origin[2]=m_Input->GetOrigin()[2];
  typename InputImageType::DirectionType direction;
  direction=this->m_Input->GetDirection();


  typedef typename InputImageType::PixelType PixelType;

  ioImage->SetRegions( region );
  ioImage->SetSpacing( spacing);
  ioImage->SetOrigin( origin);
  ioImage->SetDirection( direction);
  ioImage->Allocate();
  ioImage->FillBuffer( itk::NumericTraits< PixelType >::One);
}
}
#endif
