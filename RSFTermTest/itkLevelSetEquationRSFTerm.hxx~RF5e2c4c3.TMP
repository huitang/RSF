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

#ifndef __itkLevelSetEquationRSFTerm_hxx
#define __itkLevelSetEquationRSFTerm_hxx

#include "itkLevelSetEquationRSFTerm.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkImageFileWriter.h"

#include "itkDivideImageSetEpsilonFilter.h"
namespace itk
{

template< class TInput, class TLevelSetContainer >
LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::LevelSetEquationRSFTerm()
{
  this->m_TermName = "Chan And Vese term";
  this->m_RequiredData.insert( "Value" );
  this->m_GaussianBlurScale=1;
}

template< class TInput, class TLevelSetContainer >
LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::~LevelSetEquationRSFTerm()
{
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::Update()
{
  UpdateMeanImage();

 if (this->m_StepForSavingIntermedialResult>0&&(((int)this->currentIteration%(int)this->m_StepForSavingIntermedialResult)==0))
  {
    typedef itk::ImageFileWriter< InputImageType >     OutputWriterType;
    typename OutputWriterType::Pointer writerHeaviside = OutputWriterType::New();
    std::ostringstream filenameHS;
    filenameHS << "Foreground" << currentIteration<<".mhd";
    writerHeaviside->SetFileName(filenameHS.str().c_str());
    //writer->SetInput( binary);
    writerHeaviside->SetInput( m_ForegroundMeanImage);

    try
    {
      writerHeaviside->Update();
    }
    catch ( itk::ExceptionObject& err )
    {
      std::cout << err << std::endl;
    }
    typename OutputWriterType::Pointer writerBack= OutputWriterType::New();
    std::ostringstream filenameLS;
    filenameLS << "Background" << currentIteration<<".mhd";
    writerBack->SetFileName(filenameLS.str().c_str());
    //writer->SetInput( binary);
    writerBack->SetInput( m_BackgroundMeanImage);

    try
    {
      writerBack->Update();
    }
    catch ( itk::ExceptionObject& err )
    {
      std::cout << err << std::endl;
    }
	typename OutputWriterType::Pointer writerLevelSet= OutputWriterType::New();
	std::ostringstream filenameLevelSet;
	filenameLevelSet << "LevelSet" << currentIteration<<".mhd";
	writerLevelSet->SetFileName(filenameLevelSet.str().c_str());
	//writer->SetInput( binary);
	writerLevelSet->SetInput( m_CurrentLevelSet);

	try
	{
		writerLevelSet->Update();
	}
	catch ( itk::ExceptionObject& err )
	{
		std::cout << err << std::endl;
	}

  }

  //update m_TotalValueInternal m_TotalValueExternal m_TotalHInternal  m_TotalHExternal
std::cout<<"current iteration: "<<currentIteration<<std::endl;
currentIteration++;
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::InitializeParameters()
{ 
	currentIteration=0;
this->m_BackgroundMeanImage= InputImageType::New();
this->m_ForegroundMeanImage= InputImageType::New();
this->m_CurrentHeaviside=InputImageType::New();
this->m_CurrentHeavisideInverse=InputImageType::New();
this->m_BluredBackgroundMeanImage=InputImageType::New();
this->m_BluredBackgroundSquareMeanImage=InputImageType::New();
this->m_BluredForegroundMeanImage=InputImageType::New();
this->m_BluredForegroundSquareMeanImage=InputImageType::New();
this->m_CurrentLevelSet=InputImageType::New();

GenerateImage( this->m_BackgroundMeanImage );
GenerateImage( this->m_ForegroundMeanImage );
GenerateImage( this->m_CurrentHeaviside );
GenerateImage( this->m_CurrentHeavisideInverse );
GenerateImage( this->m_BluredBackgroundSquareMeanImage );
GenerateImage( this->m_BluredForegroundSquareMeanImage );
GenerateImage( this->m_BluredForegroundMeanImage );
GenerateImage( this->m_BluredBackgroundMeanImage );
GenerateImage( this->m_CurrentLevelSet );
  this->SetUp();
}


template< class TInput, class TLevelSetContainer >
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
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
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::UpdatePixel( const LevelSetInputIndexType& iP,
               const LevelSetOutputRealType & oldValue,
               const LevelSetOutputRealType & newValue )
{
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationRSFTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
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

    InputPixelType e1 = this->CalculateVarienceFore(iP, prodInternal);
    InputPixelType e2 = this->CalculateVarienceBack(iP, prodExternal);

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
typename LevelSetEquationRSFTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
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

    InputPixelType e1 = this->CalculateVarienceFore(iP, prodInternal);
    InputPixelType e2 = this->CalculateVarienceBack(iP, prodExternal);

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
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
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
typename LevelSetEquationRSFTerm< TInput, TLevelSetContainer >::InputPixelRealType
LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::CalculateVarienceFore(const LevelSetInputIndexType& iP, const LevelSetOutputRealType& iData)
{
  //Here calculate e1, we assume that 1K =1 see Eq.16
  InputPixelRealType intensity =
      static_cast< InputPixelRealType >( this->m_Input->GetPixel( iP ) );

  InputPixelRealType bluredForegroundMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredForegroundMeanImage->GetPixel( iP ) );

  InputPixelRealType bluredForegroundSquareMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredForegroundSquareMeanImage->GetPixel( iP ) );

  InputPixelRealType e = intensity * intensity - 2. * bluredForegroundMeanImage * intensity
      + bluredForegroundSquareMeanImage;

  return e;

}
template< class TInput, class TLevelSetContainer >
typename LevelSetEquationRSFTerm< TInput, TLevelSetContainer >::InputPixelRealType
LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::CalculateVarienceBack(const LevelSetInputIndexType& iP, const LevelSetOutputRealType& iData)
{
  //Here calculate e1, we assume that 1K =1 see Eq.16
  InputPixelRealType intensity =
      static_cast< InputPixelRealType >( this->m_Input->GetPixel( iP ) );

  InputPixelRealType bluredBackgroundMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredBackgroundMeanImage->GetPixel( iP ) );

  InputPixelRealType bluredBackgroundSquareMeanImage =
      static_cast< InputPixelRealType >( this->m_BluredBackgroundSquareMeanImage->GetPixel( iP ) );

  InputPixelRealType e = intensity * intensity - 2. * bluredBackgroundMeanImage * intensity
      + bluredBackgroundSquareMeanImage;

  return e;

}
template< class TInput, class TLevelSetContainer >
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::GenerateImage(InputImagePointer ioImage )
{
  typename InputImageType::IndexType  index = this->m_Input->GetBufferedRegion().GetIndex();
  typename InputImageType::SizeType size = this->m_Input->GetBufferedRegion().GetSize();

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
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::GetCurrentHeavisideImage()
{
  typename itk::ImageRegionIterator<  InputImageType > it( m_CurrentHeaviside, m_CurrentHeaviside->GetBufferedRegion() );
  it.GoToBegin();
  typename itk::ImageRegionIterator<  InputImageType > itInverse( m_CurrentHeavisideInverse, m_CurrentHeavisideInverse->GetBufferedRegion() );
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

     const LevelSetOutputRealType d_val2 = this->m_Heaviside->Evaluate( value ); //ARNAUD: 1 - d_val2 ??
    itInverse.Set( d_val2);
    ++itInverse;
  }
}
template< class TInput, class TLevelSetContainer >
void LevelSetEquationRSFTerm< TInput, TLevelSetContainer >
::UpdateMeanImage()
{
  typedef itk::DiscreteGaussianImageFilter<InputImageType,InputImageType> GaussianBlurFilter;
  typedef itk::MultiplyImageFilter <InputImageType,InputImageType,InputImageType> MultiplyImageType;
  typedef itk::DivideImageSetEpsilonFilter <InputImageType,InputImageType,InputImageType> DivideImageType;

  GetCurrentHeavisideImage();

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Getting current levelset used: "<<std::endl;

  typename MultiplyImageType::Pointer multiplyImageWithHeavisideImage = MultiplyImageType::New();
  multiplyImageWithHeavisideImage->SetInput1(this->m_Input);
  multiplyImageWithHeavisideImage->SetInput2(this->m_CurrentHeaviside);
  multiplyImageWithHeavisideImage->Update();
  //timeinfo = localtime ( &rawtime );
 // std::cout<< "Getting current foreground image done "<<std::endl;

  typename MultiplyImageType::Pointer multiplyImageWithHeavisideInverseImage = MultiplyImageType::New();
  multiplyImageWithHeavisideInverseImage->SetInput1(this->m_Input);
  multiplyImageWithHeavisideInverseImage->SetInput2(this->m_CurrentHeavisideInverse);
  multiplyImageWithHeavisideInverseImage->Update();

  //timeinfo = localtime ( &rawtime );
 // std::cout<< "Getting current background image done "<<std::endl;

  typename GaussianBlurFilter::Pointer gaussianblurMultiplyImageWithHeavisideImage = GaussianBlurFilter::New();
  gaussianblurMultiplyImageWithHeavisideImage->SetInput(multiplyImageWithHeavisideImage->GetOutput());
  gaussianblurMultiplyImageWithHeavisideImage->SetVariance (this->m_GaussianBlurScale);
  gaussianblurMultiplyImageWithHeavisideImage->Update();

  //timeinfo = localtime ( &rawtime );
//  std::cout<< "Blur current foreground image done "<<std::endl;

  typename GaussianBlurFilter::Pointer gaussianblurMultiplyImageWithHeavisideInverseImage = GaussianBlurFilter::New();
  gaussianblurMultiplyImageWithHeavisideInverseImage->SetInput(multiplyImageWithHeavisideInverseImage->GetOutput());
  gaussianblurMultiplyImageWithHeavisideInverseImage->SetVariance (this->m_GaussianBlurScale);
  gaussianblurMultiplyImageWithHeavisideInverseImage->Update();

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Blur current background image done "<<std::endl;

  typename GaussianBlurFilter::Pointer gaussianblurHeaviside =GaussianBlurFilter::New();
  gaussianblurHeaviside->SetInput(this->m_CurrentHeaviside);
  gaussianblurHeaviside->SetVariance(this->m_GaussianBlurScale);
  gaussianblurHeaviside->Update();

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Blur current inside heaviside image: "<<std::endl;

  typename GaussianBlurFilter::Pointer gaussianblurHeavisideInverse =GaussianBlurFilter::New();
  gaussianblurHeavisideInverse->SetInput(this->m_CurrentHeavisideInverse);
  gaussianblurHeavisideInverse->SetVariance(this->m_GaussianBlurScale);
  gaussianblurHeavisideInverse->Update();

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Blur current outside heaviside image done "<<std::endl;


  typename DivideImageType::Pointer divideImageFore =  DivideImageType::New();
  divideImageFore->SetInput1(gaussianblurMultiplyImageWithHeavisideImage->GetOutput());
  divideImageFore->SetInput2(gaussianblurHeaviside->GetOutput());
  divideImageFore->SetEpsilon(1e-10);
  divideImageFore->Update();

  typename DivideImageType::Pointer divideImageBack =  DivideImageType::New();
  divideImageBack->SetInput1(gaussianblurMultiplyImageWithHeavisideInverseImage->GetOutput());
  divideImageBack->SetInput2(gaussianblurHeavisideInverse->GetOutput());
  divideImageBack->SetEpsilon(1e-10);
  divideImageBack->Update();

  this->m_ForegroundMeanImage->Graft( divideImageFore->GetOutput() );
  this->m_BackgroundMeanImage->Graft( divideImageBack->GetOutput() );

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Get current mean images done "<<std::endl;

  typename GaussianBlurFilter::Pointer blurForegroundMeanImage= GaussianBlurFilter::New();
  blurForegroundMeanImage->SetInput(this->m_ForegroundMeanImage);
  blurForegroundMeanImage->SetVariance(m_GaussianBlurScale);
  blurForegroundMeanImage->Update();

  this->m_BluredForegroundMeanImage->Graft( blurForegroundMeanImage->GetOutput() );

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Blur current foreground mean image: "<<asctime (timeinfo)<< std::endl;

  typename GaussianBlurFilter::Pointer blurBackgroundMeanImage= GaussianBlurFilter::New();
  blurBackgroundMeanImage->SetInput(this->m_BackgroundMeanImage);
  blurBackgroundMeanImage->SetVariance(m_GaussianBlurScale);
  blurBackgroundMeanImage->Update();
  this->m_BluredBackgroundMeanImage->Graft( blurBackgroundMeanImage->GetOutput() );

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Blur current background mean image done "<<std::endl;

  typename MultiplyImageType::Pointer foregroundSquareMeanImage = MultiplyImageType::New();
  foregroundSquareMeanImage->SetInput1(this->m_ForegroundMeanImage);
  foregroundSquareMeanImage->SetInput2(this->m_ForegroundMeanImage);
  foregroundSquareMeanImage->Update();

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Get current foreground mean  square image: "<<asctime (timeinfo)<< std::endl;

  typename MultiplyImageType::Pointer backgroundSquareMeanImage = MultiplyImageType::New();
  backgroundSquareMeanImage->SetInput1(this->m_BackgroundMeanImage);
  backgroundSquareMeanImage->SetInput2(this->m_BackgroundMeanImage);
  backgroundSquareMeanImage->Update();

  //timeinfo = localtime ( &rawtime );
 // std::cout<< "Get current background mean  square image done "<<std::endl;

  typename GaussianBlurFilter::Pointer blurForegroundSquareMeanImage= GaussianBlurFilter::New();
  blurForegroundSquareMeanImage->SetInput(foregroundSquareMeanImage->GetOutput());
  blurForegroundSquareMeanImage->SetVariance(m_GaussianBlurScale);
  blurForegroundSquareMeanImage->Update();
  this->m_BluredForegroundSquareMeanImage->Graft( blurForegroundSquareMeanImage->GetOutput() );

  //timeinfo = localtime ( &rawtime );
 // std::cout<< "Blur current foreground mean  square image: "<<std::endl;

  typename GaussianBlurFilter::Pointer blurBackgroundSquareMeanImage= GaussianBlurFilter::New();
  blurBackgroundSquareMeanImage->SetInput(backgroundSquareMeanImage->GetOutput());
  blurBackgroundSquareMeanImage->SetVariance(m_GaussianBlurScale);
  blurBackgroundSquareMeanImage->Update();
  this->m_BluredBackgroundSquareMeanImage->Graft( blurBackgroundSquareMeanImage->GetOutput() );

  //timeinfo = localtime ( &rawtime );
  //std::cout<< "Blur current background mean  square image: "<<asctime (timeinfo)<< std::endl;

}
}
#endif
