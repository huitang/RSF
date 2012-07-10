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

#ifndef __itkLevelSetEquationChanAndVeseTerm_hxx
#define __itkLevelSetEquationChanAndVeseTerm_hxx

#include "itkLevelSetEquationChanAndVeseTerm.h"

namespace itk
{

template< class TInput, class TLevelSetContainer >
LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::LevelSetEquationChanAndVeseTerm() :
  m_MeanInternal( NumericTraits< InputPixelRealType >::Zero ),
  m_TotalValueInternal( NumericTraits< InputPixelRealType >::Zero ),
  m_TotalHInternal( NumericTraits< LevelSetOutputRealType >::Zero ),
  m_MeanExternal( NumericTraits< InputPixelRealType >::Zero ),
  m_TotalValueExternal( NumericTraits< InputPixelRealType >::Zero ),
  m_TotalHExternal( NumericTraits< LevelSetOutputRealType >::Zero )
{
  this->m_TermName = "Chan And Vese term";
  this->m_RequiredData.insert( "Value" );
}

template< class TInput, class TLevelSetContainer >
LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::~LevelSetEquationChanAndVeseTerm()
{
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::Update()
{
  if( this->m_TotalHInternal > NumericTraits< LevelSetOutputRealType >::epsilon() )
    {
    const LevelSetOutputRealType inv_total_hInternal = 1. / this->m_TotalHInternal;

    // depending on the pixel type, it may be more efficient to do
    // a multiplication than to do a division
    this->m_MeanInternal = this->m_TotalValueInternal * inv_total_hInternal;
    }
  else
    {
    this->m_MeanInternal = NumericTraits< InputPixelRealType >::Zero;
    }
  if( this->m_TotalHExternal > NumericTraits< LevelSetOutputRealType >::epsilon() )
  {
    const LevelSetOutputRealType inv_total_hExternal = 1. / this->m_TotalHExternal;

    // depending on the pixel type, it may be more efficient to do
    // a multiplication than to do a division
    this->m_MeanExternal = this->m_TotalValueExternal * inv_total_hExternal;
  }
  else
  {
    this->m_MeanExternal = NumericTraits< InputPixelRealType >::Zero;
  }
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::InitializeParameters()
{
  this->m_TotalValueInternal = NumericTraits< InputPixelRealType >::Zero;
  this->m_TotalHInternal = NumericTraits< LevelSetOutputRealType >::Zero;
  this->m_TotalValueExternal = NumericTraits< InputPixelRealType >::Zero;
  this->m_TotalHExternal = NumericTraits< LevelSetOutputRealType >::Zero;
  this->SetUp();
}


template< class TInput, class TLevelSetContainer >
void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
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
void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::ComputeProductInternal( const LevelSetInputIndexType& iP, LevelSetOutputRealType& prod )
{
  LevelSetOutputRealType value = this->m_CurrentLevelSetPointer->Evaluate( iP );
  prod = this->m_Heaviside->Evaluate( -value );

}

  template< class TInput, class TLevelSetContainer >
  void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
    ::ComputeProductExternal( const LevelSetInputIndexType& iP, LevelSetOutputRealType& prod )
  {
    this->ComputeProductTermExternal( iP, prod );
    LevelSetPointer levelSet = this->m_LevelSetContainer->GetLevelSet( this->m_CurrentLevelSetId );
    LevelSetOutputRealType value = levelSet->Evaluate( iP );
    prod *= -(1 - this->m_Heaviside->Evaluate( -value ) );
}
  template< class TInput, class TLevelSetContainer >
  void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
    ::ComputeProductTermExternal( const LevelSetInputIndexType& iP, LevelSetOutputRealType& prod )
  {
    prod = -1 * NumericTraits< LevelSetOutputRealType >::One;

    if( this->m_LevelSetContainer->HasDomainMap() )
    {
      DomainMapImageFilterType * domainMapFilter = this->m_LevelSetContainer->GetDomainMapFilter();
      CacheImageType * cacheImage = domainMapFilter->GetOutput();
      const LevelSetIdentifierType id = cacheImage->GetPixel( iP );

      typedef typename DomainMapImageFilterType::DomainMapType DomainMapType;
      const DomainMapType domainMap = domainMapFilter->GetDomainMap();
      typename DomainMapType::const_iterator levelSetMapItr = domainMap.find(id);

      if( levelSetMapItr != domainMap.end() )
      {
        const IdListType * idList = levelSetMapItr->second.GetIdList();

        LevelSetIdentifierType kk;
        LevelSetPointer levelSet;
        LevelSetOutputRealType value;

        IdListConstIterator idListIt = idList->begin();
        while( idListIt != idList->end() )
        {
          //! \todo Fix me for string identifiers
          kk = *idListIt - 1;
          if( kk != this->m_CurrentLevelSetId )
          {
            levelSet = this->m_LevelSetContainer->GetLevelSet( kk );
            value = levelSet->Evaluate( iP );
            prod *= ( NumericTraits< LevelSetOutputRealType >::One - this->m_Heaviside->Evaluate( -value ) );
          }
          ++idListIt;
        }
      }
    }
    else
    {
      LevelSetIdentifierType kk;
      LevelSetPointer levelSet;
      LevelSetOutputRealType value;

      typename LevelSetContainerType::Iterator lsIt = this->m_LevelSetContainer->Begin();

      while( lsIt != this->m_LevelSetContainer->End() )
      {
        kk = lsIt->GetIdentifier();
        if( kk != this->m_CurrentLevelSetId )
        {
          levelSet = this->m_LevelSetContainer->GetLevelSet( kk );
          value = levelSet->Evaluate( iP );
          prod *= ( NumericTraits< LevelSetOutputRealType >::One - this->m_Heaviside->Evaluate( -value ) );
        }
        ++lsIt;
      }
    }
  }

template< class TInput, class TLevelSetContainer >
void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::UpdatePixel( const LevelSetInputIndexType& iP,
               const LevelSetOutputRealType & oldValue,
               const LevelSetOutputRealType & newValue )
{
  // For each affected h val: h val = new hval (this will dirty some cvals)
  InputPixelType input = this->m_Input->GetPixel( iP );

  const LevelSetOutputRealType oldH = this->m_Heaviside->Evaluate( -oldValue );
  const LevelSetOutputRealType newH = this->m_Heaviside->Evaluate( -newValue );
  const LevelSetOutputRealType changeInternal = newH - oldH;

  // update the foreground constant for current level-set function
  this->m_TotalHInternal += changeInternal ;
  this->m_TotalValueInternal += input * changeInternal ;

  const LevelSetOutputRealType changeExternal = oldH - newH;//(1 - newH) - (1 - oldH);

  // Determine the change in the product factor
    LevelSetOutputRealType prodExternal;
    this->ComputeProductTermExternal( iP, prodExternal);
  const LevelSetOutputRealType productChangeExternal = -( prodExternal * changeExternal );

  this->m_TotalHExternal += productChangeExternal;
  this->m_TotalValueExternal  += input * productChangeExternal;

}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::Value( const LevelSetInputIndexType& iP )
{
  if( this->m_Heaviside.IsNotNull() )
    {
    const LevelSetOutputRealType value =
      static_cast< LevelSetOutputRealType >( this->m_CurrentLevelSetPointer->Evaluate( iP ) );

    const LevelSetOutputRealType d_val = this->m_Heaviside->EvaluateDerivative( -value );

    const InputPixelType pixel = this->m_Input->GetPixel( iP );
  LevelSetOutputRealType prodInternal = 1;
  LevelSetOutputRealType prodExternal = 1;

  this->ComputeProductTermInternal( iP, prodInternal );
this->ComputeProductTermExternal( iP, prodExternal );

    const LevelSetOutputRealType oValue = d_val *
      static_cast< LevelSetOutputRealType >( prodInternal *this->m_InternalCoefficient*( pixel - this->m_MeanInternal ) * ( pixel - this->m_MeanInternal ) + prodExternal *this->m_ExternalCoefficient*( pixel - this->m_MeanExternal ) * ( pixel - this->m_MeanExternal ));

    return oValue;
    }
  else
    {
    itkWarningMacro( << "m_Heaviside is NULL" );
    }
  return NumericTraits< LevelSetOutputPixelType >::Zero;
}

template< class TInput, class TLevelSetContainer >
typename LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >::LevelSetOutputRealType
LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::Value( const LevelSetInputIndexType& iP, const LevelSetDataType& iData )
{
  if( this->m_Heaviside.IsNotNull() )
    {
    const LevelSetOutputRealType value = iData.Value.m_Value;

    const LevelSetOutputRealType d_val = this->m_Heaviside->EvaluateDerivative( -value );

    const InputPixelType pixel = this->m_Input->GetPixel( iP );
  LevelSetOutputRealType prodInternal = 1;
  LevelSetOutputRealType prodExternal = 1;

  this->ComputeProductTermInternal( iP, prodInternal );
  this->ComputeProductTermExternal( iP, prodExternal );

  const LevelSetOutputRealType oValue = d_val *
    static_cast< LevelSetOutputRealType >( prodInternal *this->m_InternalCoefficient*( pixel - this->m_MeanInternal ) * ( pixel - this->m_MeanInternal ) +  prodExternal *this->m_ExternalCoefficient*( pixel - this->m_MeanExternal ) * ( pixel - this->m_MeanExternal ));

    return oValue;
    }
  else
    {
    itkWarningMacro( << "m_Heaviside is NULL" );
    }
  return NumericTraits< LevelSetOutputPixelType >::Zero;
}

template< class TInput, class TLevelSetContainer >
void LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
::Accumulate( const InputPixelType& iPix, const LevelSetOutputRealType& iHIn, const LevelSetOutputRealType& iHEx )
{
  this->m_TotalValueInternal += static_cast< InputPixelRealType >( iPix ) *
      static_cast< LevelSetOutputRealType >( iHIn );
  this->m_TotalHInternal += static_cast< LevelSetOutputRealType >( iHIn );
  this->m_TotalValueExternal += static_cast< InputPixelRealType >( iPix ) *
    static_cast< LevelSetOutputRealType >( iHEx);
  this->m_TotalHExternal += static_cast< LevelSetOutputRealType >( iHEx );
}

}
#endif
