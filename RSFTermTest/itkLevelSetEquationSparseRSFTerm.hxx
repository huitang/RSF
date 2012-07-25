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
 *  Author: Hui Tang
 *=========================================================================*/

#ifndef __itkLevelSetEquationSparseRSFTerm_hxx
#define __itkLevelSetEquationSparseRSFTerm_hxx

#include "itkLevelSetEquationSparseRSFTerm.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkImageFileWriter.h"

#include "itkDivideImageSetEpsilonFilter.h"
namespace itk
{

template< class TInput, class TLevelSetContainer >
LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::LevelSetEquationSparseRSFTerm()
{
  this->m_TermName = "Sparse RSF term";
  this->m_RequiredData.insert( "Value" );
  this->m_GaussianBlurScale=1;
}

template< class TInput, class TLevelSetContainer >
LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::~LevelSetEquationSparseRSFTerm()
{
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::Update()
{
  UpdateMeanImage();
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::InitializeParameters()
{
  this->m_BackgroundMeanImage= InputImageType::New();
  this->m_ForegroundMeanImage= InputImageType::New();
  this->m_CurrentHeaviside=InputImageType::New();
  this->m_CurrentInverseHeaviside=InputImageType::New();
  this->m_BluredBackgroundMeanImage=InputImageType::New();
  this->m_BluredBackgroundSquareMeanImage=InputImageType::New();
  this->m_BluredForegroundMeanImage=InputImageType::New();
  this->m_BluredForegroundSquareMeanImage=InputImageType::New();
  this->m_CurrentLevelSet=InputImageType::New();

  GenerateImage( this->m_BackgroundMeanImage );
  GenerateImage( this->m_ForegroundMeanImage );
  GenerateImage( this->m_CurrentHeaviside );
  GenerateImage( this->m_CurrentInverseHeaviside );
  GenerateImage( this->m_BluredBackgroundSquareMeanImage );
  GenerateImage( this->m_BluredForegroundSquareMeanImage );
  GenerateImage( this->m_BluredForegroundMeanImage );
  GenerateImage( this->m_BluredBackgroundMeanImage );
  GenerateImage( this->m_CurrentLevelSet );

  this->SetUp();
}


template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::Initialize( const LevelSetInputIndexType& iP )
{
  if( this->m_Heaviside.IsNotNull() )
    {
    InputPixelType pixel = this->m_Input->GetPixel( iP );

    LevelSetOutputRealType prodInternal;
    this->ComputeProductInternal( iP, prodInternal );

    LevelSetOutputRealType prodExternal;
    this->ComputeProductExternal( iP, prodExternal );

    this->Accumulate( pixel, prodInternal,prodExternal );
    }
  else
    {
    itkWarningMacro( << "m_Heaviside is NULL" );
    }
}


template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::UpdatePixel( const LevelSetInputIndexType& iP,
               const LevelSetOutputRealType & oldValue,
               const LevelSetOutputRealType & newValue )
{
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::Value( const LevelSetInputIndexType& iP )
{
  if( this->m_Heaviside.IsNotNull() )
    {
    const LevelSetOutputRealType value =
      static_cast< LevelSetOutputRealType >( this->m_CurrentLevelSetPointer->Evaluate( iP ) );

    const LevelSetOutputRealType d_val = this->m_Heaviside->EvaluateDerivative( -value );

    LevelSetOutputRealType prodInternal = 1;
    LevelSetOutputRealType prodExternal = 1;

    this->ComputeProductTermInternal( iP, prodInternal );
    this->ComputeProductTermExternal( iP, prodExternal );

    InputPixelType e1 = this->CalculateVarianceForeground(iP, prodInternal);
    InputPixelType e2 = this->CalculateVarianceBackground(iP, prodExternal);

    const LevelSetOutputRealType oValue = d_val *
      static_cast< LevelSetOutputRealType >( prodInternal *this->m_InternalCoefficient*e1 +  prodExternal *this->m_ExternalCoefficient*e2);

    return oValue;
    }
  else
    {
    itkWarningMacro( << "m_Heaviside is NULL" );
    }
  return NumericTraits< LevelSetOutputPixelType >::Zero;
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::Value( const LevelSetInputIndexType& iP, const LevelSetDataType& iData )
{
  if( this->m_Heaviside.IsNotNull() )
    {
    const LevelSetOutputRealType value = iData.Value.m_Value;

    const LevelSetOutputRealType d_val = this->m_Heaviside->EvaluateDerivative( -value );

    LevelSetOutputRealType prodInternal = 1;
    LevelSetOutputRealType prodExternal = 1;

    this->ComputeProductTermInternal( iP, prodInternal ); // prodInternal = 1
    this->ComputeProductTermExternal( iP, prodExternal );

    InputPixelType e1 = this->CalculateVarianceForeground(iP, prodInternal);
    InputPixelType e2 = this->CalculateVarianceBackground(iP, prodExternal);

    const LevelSetOutputRealType oValue = d_val *
      static_cast< LevelSetOutputRealType >( prodInternal * this->m_InternalCoefficient * e1 +
                                             prodExternal * this->m_ExternalCoefficient * e2 );

    return oValue;
    }
  else
    {
    itkWarningMacro( << "m_Heaviside is NULL" );
    }
  return NumericTraits< LevelSetOutputPixelType >::Zero;
}


template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::Accumulate( const InputPixelType& iPix, const LevelSetOutputRealType& iHIn, const LevelSetOutputRealType& iHEx )
{
  /*this->m_TotalValueInternal += static_cast< InputPixelRealType >( iPix ) *
      static_cast< LevelSetOutputRealType >( iHIn );
  this->m_TotalHInternal += static_cast< LevelSetOutputRealType >( iHIn );
  this->m_TotalValueExternal += static_cast< InputPixelRealType >( iPix ) *
    static_cast< LevelSetOutputRealType >( iHEx);
  this->m_TotalHExternal += static_cast< LevelSetOutputRealType >( iHEx );*/
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >::InputPixelRealType
LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::CalculateVarianceForeground(const LevelSetInputIndexType& iP, const LevelSetOutputRealType& iData)
{
  //Here calculate e1, we assume that 1K =1 see Eq.16
 const InputPixelRealType intensity =
      static_cast< InputPixelRealType >( this->m_Input->GetPixel( iP ) );

 const InputPixelRealType bluredForegroundMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredForegroundMeanImage->GetPixel( iP ) );

 const InputPixelRealType bluredForegroundSquareMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredForegroundSquareMeanImage->GetPixel( iP ) );

  InputPixelRealType e = intensity * intensity - 2. * bluredForegroundMeanImage * intensity
      + bluredForegroundSquareMeanImage;

  return e;
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >::InputPixelRealType
LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::CalculateVarianceBackground(const LevelSetInputIndexType& iP, const LevelSetOutputRealType& iData)
{
  //Here calculate e1, we assume that 1K =1 see Eq.16
  const InputPixelRealType intensity =
      static_cast< InputPixelRealType >( this->m_Input->GetPixel( iP ) );

  const InputPixelRealType bluredBackgroundMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredBackgroundMeanImage->GetPixel( iP ) );

  const InputPixelRealType bluredBackgroundSquareMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredBackgroundSquareMeanImage->GetPixel( iP ) );

  InputPixelRealType e = intensity * intensity - 2. * bluredBackgroundMeanImage * intensity
      + bluredBackgroundSquareMeanImage;

  return e;
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::GenerateImage(InputImagePointer ioImage )
{
  typename InputImageType::IndexType  index = this->m_Input->GetBufferedRegion().GetIndex();
  typename InputImageType::SizeType   size  = this->m_Input->GetBufferedRegion().GetSize();

  typename InputImageType::RegionType region;
  region.SetIndex( index );
  region.SetSize( size );

  typename InputImageType::SpacingType spacing;
  spacing=this->m_Input->GetSpacing();

  typename InputImageType::PointType origin;
  origin=this->m_Input->GetOrigin();

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

template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::GetCurrentHeavisideImage()
{
  typename itk::ImageRegionIterator<  InputImageType > it( m_CurrentHeaviside, m_CurrentHeaviside->GetBufferedRegion() );
  it.GoToBegin();

  typename itk::ImageRegionIterator<  InputImageType > itInverse( m_CurrentInverseHeaviside, m_CurrentInverseHeaviside->GetBufferedRegion() );
  itInverse.GoToBegin();

  typename itk::ImageRegionIterator<  InputImageType > itLevel( m_CurrentLevelSet, m_CurrentLevelSet->GetBufferedRegion() );
  itLevel.GoToBegin();

  while( !it.IsAtEnd())
    {
    const LevelSetOutputRealType value =static_cast< LevelSetOutputRealType >( this->m_CurrentLevelSetPointer->Evaluate( it.GetIndex()  ) );
    itLevel.Set(value);
    ++itLevel;
    //LevelSetPointer levelSet = this->m_LevelSetContainer->GetLevelSet( this->m_CurrentLevelSetId );
    //LevelSetOutputRealType value = levelSet->Evaluate( it.GetIndex()  );
    const LevelSetOutputRealType d_val1 = this->m_Heaviside->Evaluate( -value );
    it.Set( d_val1);
    ++it;

    itInverse.Set( 1-d_val1);
    ++itInverse;
    }
}
template< class TInput, class TLevelSetContainer >
void LevelSetEquationSparseRSFTerm< TInput, TLevelSetContainer >
::UpdateMeanImage(){
	typedef itk::DiscreteGaussianImageFilter<InputImageType,InputImageType>
		GaussianBlurFilterType;
	typedef itk::MultiplyImageFilter <InputImageType,InputImageType,InputImageType>
		MultiplyImageFilterType;
	typedef itk::DivideImageSetEpsilonFilter <InputImageType,InputImageType,InputImageType> 
		DivideImageFilterType;

	GetCurrentHeavisideImage();

	typename MultiplyImageFilterType::Pointer multiplyImageWithHeavisideImage =
		MultiplyImageFilterType::New();
	multiplyImageWithHeavisideImage->SetInput1(this->m_Input);
	multiplyImageWithHeavisideImage->SetInput2(this->m_CurrentHeaviside);
	multiplyImageWithHeavisideImage->Update();

	typename MultiplyImageFilterType::Pointer multiplyImageWithInverseHeavisideImage =
		MultiplyImageFilterType::New();
	multiplyImageWithInverseHeavisideImage->SetInput1(this->m_Input);
	multiplyImageWithInverseHeavisideImage->SetInput2(this->m_CurrentInverseHeaviside);
	multiplyImageWithInverseHeavisideImage->Update();


	typename GaussianBlurFilterType::Pointer gaussianBlurMultiplyImageWithHeavisideImage =
		GaussianBlurFilterType::New();
	gaussianBlurMultiplyImageWithHeavisideImage->SetInput(
		multiplyImageWithHeavisideImage->GetOutput());
	gaussianBlurMultiplyImageWithHeavisideImage->SetVariance(this->m_GaussianBlurScale);
	gaussianBlurMultiplyImageWithHeavisideImage->Update();

	typename GaussianBlurFilterType::Pointer gaussianBlurMultiplyImageWithInverseHeavisideImage = 
		GaussianBlurFilterType::New();
	gaussianBlurMultiplyImageWithInverseHeavisideImage->SetInput(
		multiplyImageWithInverseHeavisideImage->GetOutput());
	gaussianBlurMultiplyImageWithInverseHeavisideImage->SetVariance(this->m_GaussianBlurScale);
	gaussianBlurMultiplyImageWithInverseHeavisideImage->Update();

	typename GaussianBlurFilterType::Pointer gaussianBlurHeaviside =
		GaussianBlurFilterType::New();
	gaussianBlurHeaviside->SetInput(this->m_CurrentHeaviside);
	gaussianBlurHeaviside->SetVariance(this->m_GaussianBlurScale);
	gaussianBlurHeaviside->Update();

	typename GaussianBlurFilterType::Pointer gaussianBlurInverseHeaviside =
		GaussianBlurFilterType::New();
	gaussianBlurInverseHeaviside->SetInput(this->m_CurrentInverseHeaviside);
	gaussianBlurInverseHeaviside->SetVariance(this->m_GaussianBlurScale);
	gaussianBlurInverseHeaviside->Update();

	typename DivideImageFilterType::Pointer divideImageForeground = DivideImageFilterType::New();
	divideImageForeground->SetInput1(
		gaussianBlurMultiplyImageWithHeavisideImage->GetOutput());
	divideImageForeground->SetInput2( gaussianBlurHeaviside->GetOutput());
	divideImageForeground->SetEpsilon(1e-10);
	divideImageForeground->Update();

	typename DivideImageFilterType::Pointer divideImageBackground =                                 DivideImageFilterType::New();
	divideImageBackground->SetInput1(
		gaussianBlurMultiplyImageWithInverseHeavisideImage->GetOutput());
	divideImageBackground->SetInput2(
		gaussianBlurInverseHeaviside->GetOutput());
	divideImageBackground->SetEpsilon(1e-10);
	divideImageBackground->Update();

	typename GaussianBlurFilterType::Pointer blurForegroundMeanImage =   
		GaussianBlurFilterType::New();
	blurForegroundMeanImage->SetInput(this->m_ForegroundMeanImage);
	blurForegroundMeanImage->SetVariance(m_GaussianBlurScale);
	blurForegroundMeanImage->Update();

	this->m_BluredForegroundMeanImage->Graft(blurForegroundMeanImage->GetOutput() );

	typename GaussianBlurFilterType::Pointer blurBackgroundMeanImage= 
		GaussianBlurFilterType::New();
	blurBackgroundMeanImage->SetInput(this->m_BackgroundMeanImage);
	blurBackgroundMeanImage->SetVariance(m_GaussianBlurScale);
	blurBackgroundMeanImage->Update();
	this->m_BluredBackgroundMeanImage->Graft(blurBackgroundMeanImage->GetOutput() );

	typename MultiplyImageFilterType::Pointer foregroundSquareMeanImage =
		MultiplyImageFilterType::New();
	foregroundSquareMeanImage->SetInput1(this->m_ForegroundMeanImage);
	foregroundSquareMeanImage->SetInput2(this->m_ForegroundMeanImage);
	foregroundSquareMeanImage->Update();

	typename MultiplyImageFilterType::Pointer backgroundSquareMeanImage =
		MultiplyImageFilterType::New();
	backgroundSquareMeanImage->SetInput1(this->m_BackgroundMeanImage);
	backgroundSquareMeanImage->SetInput2(this->m_BackgroundMeanImage);
	backgroundSquareMeanImage->Update();

	typename GaussianBlurFilterType::Pointer blurForegroundSquareMeanImage= 
		GaussianBlurFilterType::New();
	blurForegroundSquareMeanImage->SetInput(foregroundSquareMeanImage->GetOutput());
	blurForegroundSquareMeanImage->SetVariance(m_GaussianBlurScale);
	blurForegroundSquareMeanImage->Update();
	this->m_BluredForegroundSquareMeanImage->Graft( 
		blurForegroundSquareMeanImage->GetOutput() );

	typename GaussianBlurFilterType::Pointer blurBackgroundSquareMeanImage= 
		GaussianBlurFilterType::New();
	blurBackgroundSquareMeanImage->SetInput(backgroundSquareMeanImage->GetOutput());
	blurBackgroundSquareMeanImage->SetVariance(m_GaussianBlurScale);
	blurBackgroundSquareMeanImage->Update();
	this->m_BluredBackgroundSquareMeanImage->Graft( 
		blurBackgroundSquareMeanImage->GetOutput() );
}
}
#endif
